import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM
from tqdm import tqdm, trange
import gzip 
import json
import os
import argparse
import numpy as np
import fnmatch
from typing import DefaultDict
import sys
import pickle
from datasets import load_dataset

#largest corpus = 	5.42Mb
#beirname = ["scifact","arguana","trec-covid","nfcorpus","nq","hotpotqa","fiqa","webis-touche2020","quora","dbpedia-entity","scidocs","fever","climate-fever"]
#beirname = ["arguana","trec-covid"]
parser = argparse.ArgumentParser()
agg = "max"
parser.add_argument('--checkpoint',type=str,help="checkpoint path if needed, no necessary",default="")
parser.add_argument('--threshold',type=int,default=0,help="indexing threshold")
parser.add_argument('--output',type=str,help="output prefix")
parser.add_argument('--scale',type=int,help='score scale of the weight',default=100)
parser.add_argument('--model',type=str)
parser.add_argument('--beirname',type=str)
args = parser.parse_args()

out_dir = args.output
beirname = [args.beirname]
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# loading model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
# Load model directly


tokenizer = AutoTokenizer.from_pretrained(args.model)
model = BertForMaskedLM.from_pretrained(args.model).to(device)
model.eval()
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

scale = args.scale
file_per = 100000
i = 0
#beirname = ["arguana","fiqa","nfcorpus","quora","scidocs","scifact","trec-covid" "webis-touche2020" "climate-fever" "dbpedia-entity" "fever" "hotpotqa" "nq"]
#beirname = ["hotpotqa"]*2
fo = None
#skip msmarco
for dsname in beirname:
    print(f"encoding {dsname} queries")
    total_len = 0
    batchsz = 128 # batchsz has to be able to divide file_per
    dataset = load_dataset(f"BeIR/{dsname}-qrels")["test"]
    test_qids = list(map(str,list(dataset['query-id'])))
    #print(f"{dsname} approximate queries number: {len(test_qids)}")
    dataset = load_dataset(f"BeIR/{dsname}","queries")["queries"]
    repllama_embds = DefaultDict()
    fo = open(os.path.join(out_dir, f"{dsname}_queries.pisa.dev"), "w")
    for q in tqdm(dataset,desc=f"{dsname}"):
        with torch.inference_mode():
            if q["_id"] not in test_qids:
                continue
            qid = q["_id"]
            query_text = q['text']
            tokens = tokenizer([query_text], return_tensors='pt', padding=True, truncation=True,max_length=256).to(device)
            doc_rep = model(**tokens)["logits"]#.squeeze()  # (sparse) doc rep in voc space, shape (30522,)
            doc_rep = torch.max(torch.log(1 + torch.relu(doc_rep)) * tokens.attention_mask.unsqueeze(-1), dim=1).values
            # get the number of non-zero dimensions in the rep:
            doc_rep = doc_rep.squeeze()
            #print(doc_rep[torch.nonzero(doc_rep).squeeze()])
            col = torch.nonzero(doc_rep).squeeze().detach().cpu().tolist()
            #print(col.size())
            #weights = doc_rep[col].cpu().tolist()
            #d = {reverse_voc[k]: int(v * scale) for k, v in zip(col, weights)}
            #col = torch.nonzero(doc_rep).squeeze().detach().cpu().tolist()
            #print(col.size())
            total_len += len(col)
            weights = doc_rep[col].cpu().numpy()
            weights = np.clip(np.rint(weights * scale),0,255).astype(int).tolist()
            d = {reverse_voc[k]: int(v) for k, v in zip(col, weights)}
            q = []
            for tok in d:
                for _ in range(d[tok]):
                    q.append(tok)
            #outline = json.dumps({"id": int(did), "content": doc, "vector": d}) + "\n"
            fo.write(f"{qid}: {' '.join(q)}\n")
    print(f"{dsname} average query length: {total_len/len(dataset)}")
    fo.close()
