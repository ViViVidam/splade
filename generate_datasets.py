import json
from collections import defaultdict
from transformers import AutoTokenizer, BertForMaskedLM, LlamaModel
import numpy as np
import tqdm
import torch
import pickle
def read_msmarco_corpus(fIn):
    pid_to_doc = {}
    for line in fIn:
        pid, text = line.strip().split("\t")
        pid_to_doc[pid] = text
    return pid_to_doc 
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

def read_qrels(fIn) -> dict:
    qrels = defaultdict(str)
    for line in fIn:
        qid,_,doc,rel = line.strip().split('\t')
        assert(int(rel)==1)
        qrels[qid] = doc
    return qrels

def read_queries(fIn) -> dict:
    queries = defaultdict(str)
    for line in fIn:
        qid,text = line.strip().split('\t')
        # assert(int(rel)==1)
        queries[qid] = text
    return queries

if __name__ == "__main__":
    model_name = "naver/splade-cocondenser-selfdistil"
    #model = BertForMaskedLM.from_pretrained(model_name)
    #model = Wrapper(model=model)
    #model.to('cuda')
    #model.eval()
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open("/projects/bcgk/yzound/datasets/msmarco/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl","rb") as file:
        ce_score = pickle.load(file)
    with open("/projects/bcgk/yzound/datasets/msmarco/qrels.train.tsv","r") as reader:
        qrel = read_qrels(reader)
    with open("/projects/bcgk/zwang48/sclr/msmarco-full/collection.tsv","r") as reader:
        corpus = read_msmarco_corpus(reader)
    with open("/projects/bcgk/yzound/datasets/msmarco/queries.train.tsv","r") as reader:
        queries = read_queries(reader)
    with open("/projects/bfcj/yzound/index/splade_selfdistil/run.json") as reader:
        run = json.load(reader)
    with open("/projects/bfcj/yzound/index/splade_selfdistil/splade_selfdistill_hn.jsonl","w+") as fout:
        for qid,retrieved in tqdm.tqdm(run.items()):
            if qid not in qrel.keys():
                continue
            rel = qrel[qid]
            dids = []
            scores = []
            for k,v in retrieved.items():
                if k == rel:
                    pos_score = v
                else:
                    dids.append(k)
                    scores.append(v)
            dids = np.array(dids)
            scores = np.array(scores)
            idx = np.argsort(scores)[::-1]
            neg_pids = dids[idx[:100]]
            neg_scores = [ce_score[qid][str(did)] for did in dids[idx[:100]]]
            outitem = {"question":queries[qid],"pos_pid":rel,"neg_pids":dids[idx[:100]],"pos_score":ce_score[qid][rel],"neg_scores":neg_scores}
            outline = json.dumps(outitem)+"\n"
            fout.write(outline.encode('utf-8'))