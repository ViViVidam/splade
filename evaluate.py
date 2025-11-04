from transformers import AutoTokenizer, BertForMaskedLM
import argparse
from torch.utils.data import DataLoader, Dataset
import os
from task.evaluator import SparseRetrieval
import torch
import ujson
from pytrec_eval import RelevanceEvaluator, supported_measures
from collections import Counter
import json
def mrr_k(run, qrel, k, agg=True):
    evaluator = RelevanceEvaluator(qrel, {"recip_rank"})
    #truncated = truncate_run(run, k)
    
    mrr = evaluator.evaluate(run)
    if agg:
        mrr = sum([d["recip_rank"] for d in mrr.values()]) / max(1, len(mrr))
    return mrr

def recall_k(run, qrel, k, agg=True):
    evaluator = RelevanceEvaluator(qrel, {"recall"})
    out_eval = evaluator.evaluate(run)

    total_v = 0.
    included_k = f"recall_{k}"
    for q, v_dict in out_eval.items():
        for k, v in v_dict.items():
            if k == included_k:
                total_v += v 

    return total_v / len(out_eval)            

def evaluate(run, qrel, metric, agg=True, select=None):
    assert metric in supported_measures, print("provide valid pytrec_eval metric")
    evaluator = RelevanceEvaluator(qrel, {metric})
    out_eval = evaluator.evaluate(run)
    res = Counter({})
    if agg:
        for d in out_eval.values():  # when there are several results provided (e.g. several cut values)
            res += Counter(d)
        res = {k: v / len(out_eval) for k, v in res.items()}
        if select is not None:
            string_dict = "{}_{}".format(metric, select)
            if string_dict in res:
                return res[string_dict]
            else:  # If the metric is not on the dict, say that it was 0
                return 0
        else:
            return res
    else:
        return out_eval
    
def load_and_evaluate(run_file_path, metric,qrel_file_path="/projects/bcgk/zwang48/sclr/msmarco-full/dev_queries/dev_qrel.json"):
    with open(qrel_file_path) as reader:
        qrel = json.load(reader)
    with open(run_file_path) as reader:
        run = json.load(reader)
    # for trec, qrel_binary.json should be used for recall etc., qrel.json for NDCG.
    # if qrel.json is used for binary metrics, the binary 'splits' are not correct
    if "TREC" in qrel_file_path:
        assert ("binary" not in qrel_file_path) == (metric == "ndcg" or metric == "ndcg_cut"), (qrel_file_path, metric)
    if metric == "mrr_10":
        res = mrr_k(run, qrel, k=10)
        print("MRR@10:", res)
        return {"mrr_10": res}
    else:
        res = evaluate(run, qrel, metric=metric)
        print(metric, "==>", res)
        return res

def evaluate_msmarco(eval_run_path):
    res = {}
    for metric in ["mrr_10","recall"]:
        metric_val = load_and_evaluate(eval_run_path, metric)
        res[metric] = metric_val
    os.makedirs(args.out_path, exist_ok=True)
    with open(os.path.join(args.out_path, "perf.json"), "w") as fout:
        ujson.dump(res, fout, indent=4)

def read_msmarco_query(query_path):
    qid_to_query = {}
    with open(query_path) as fin:
        for line in fin:
            qid, query = line.strip().split("\t")
            qid_to_query[qid] = query
    return qid_to_query

class MSMARCOQueryDataset(Dataset):
    def __init__(self, query_path):
        # query and corpus sets have same format in msmarco
        self.qid_to_query = read_msmarco_query(query_path)
        self.qids = list(self.qid_to_query.keys())
        
    def __len__(self):
        return len(self.qid_to_query)
    
    def __getitem__(self, idx):
        qid = self.qids[idx]
        query = self.qid_to_query[qid]
        
        return qid, query

class Collector:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        ids, texts = [list(xs) for xs in zip(*batch)]
        #print(texts)
        tokenized_contexts = self.tokenizer(texts,
                                            max_length=self.max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        return {
            **{k: v for k, v in tokenized_contexts.items()},
            "id": ids,
            "texts": texts
        }

class Wrapper(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
        self.vocab_size = model.config.vocab_size
        self.config = model.config
    def forward(self,q_kwargs=None,d_kwargs=None):
        if q_kwargs is not None:
            feature = self.model(**q_kwargs)["logits"]
            rep =torch.log(torch.relu(torch.max(feature + ( 1 - q_kwargs["attention_mask"].unsqueeze(-1)) * -1e6, dim=1)[0]) + 1)
            return {'q_rep': rep}
        if d_kwargs is not None:
            feature = self.model(**d_kwargs)["logits"]
            rep =torch.log(torch.relu(torch.max(feature + ( 1 - d_kwargs["attention_mask"].unsqueeze(-1)) * -1e6, dim=1)[0]) + 1)
            return {'d_rep': rep}    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str, required=True)
    parser.add_argument('--max_length',default=256)
    parser.add_argument('--eval_batch_size',default=1)
    parser.add_argument('--query_path',default="/projects/bcgk/zwang48/sclr/msmarco-full/collection.tsv")
    parser.add_argument('--index_dir',type=str, required=True)
    parser.add_argument('--out_path',default=None)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    q_collection = MSMARCOQueryDataset(args.query_path)
    model = Wrapper(model=BertForMaskedLM.from_pretrained(args.model))
    model.eval()
    q_collator = Collector(tokenizer=tokenizer,max_length=args.max_length)

    q_loader = DataLoader(q_collection, batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=q_collator)
    if args.out_path is None:
        args.out_path = args.index_dir
    os.makedirs(args.out_path, exist_ok=True)
    config = {
        "index_dir": args.index_dir,
        "out_dir": args.out_path
    }
    #evaluate_msmarco(os.path.join(args.out_path,"run.json"))
    retriever = SparseRetrieval(config=config, model=model, compute_stats=True, 
                                dim_voc=model.vocab_size, device='cuda')
    retriever.retrieve(q_loader, top_k=1000, threshold=0.0)
    evaluate_msmarco(os.path.join(args.out_path,"run.json"))