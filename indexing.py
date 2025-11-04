from task.evaluator import SparseIndexing, SparseRetrieval
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForMaskedLM
import argparse

def read_msmarco_corpus(corpus_path):
    pid_to_doc = {}
    with open(corpus_path) as fin:
        for line in fin:
            pid, text = line.strip().split("\t")
            pid_to_doc[pid] = (None, text)
    return pid_to_doc 

def read_msmarco_query(query_path):
    qid_to_query = {}
    with open(query_path) as fin:
        for line in fin:
            qid, query = line.strip().split("\t")
            qid_to_query[qid] = query
    return qid_to_query

def get_doc_text(title, text):
        if title is None:
            return text
        else:
            return f"title: {title} | context: {text}"

class CollectionDataset(Dataset):
    def __init__(self, corpus_path, data_source=None):
        self.pid_to_doc = read_msmarco_corpus(corpus_path)
        self.pids = list(self.pid_to_doc.keys())
        
    def __len__(self):
        return len(self.pids)
    
    def __getitem__(self, idx):
        pid = self.pids[idx]
        text = get_doc_text(*self.pid_to_doc[pid])
        
        return pid, text


class Collector:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        ids, texts = [list(xs) for xs in zip(*batch)]
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
    parser.add_argument('--eval_batch_size',default=128)
    parser.add_argument('--corpus_path',default="/projects/bcgk/zwang48/sclr/msmarco-full/collection.tsv")
    parser.add_argument('--index_dir',type=str, required=True)
    args = parser.parse_args()
    d_collection = CollectionDataset(corpus_path=args.corpus_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BertForMaskedLM.from_pretrained(args.model)
    model = Wrapper(model=model)
    model.eval()
    d_collator = Collector(tokenizer=tokenizer,max_length=args.max_length)
    d_loader = DataLoader(d_collection, batch_size=args.eval_batch_size, shuffle=False, 
                          collate_fn=d_collator)
    #if torch.distributed.get_world_size() > 1:
    #    index_dir = args.index_dir[:-1] if args.index_dir.endswith("/") else args.index_dir
    #    index_dir = f"{index_dir}_{torch.distributed.get_rank()}"
    #else:
    index_dir = args.index_dir
    #print(index_dir, args.local_rank, model.vocab_size)
    indexer = SparseIndexing(model, tokenizer=tokenizer,index_dir=index_dir, compute_stats=True, dim_voc=model.vocab_size,
                            device='cuda')
    indexer.index(d_loader)