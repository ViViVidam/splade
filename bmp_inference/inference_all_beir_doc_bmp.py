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

#largest corpus = 	5.42M
#beirname = ["arguana","scifact","trec-covid","nfcorpus","fiqa","webis-touche2020","quora","scidocs"]
#beirname = ["fever"]#,"climate-fever","dbpedia-entity"]
#beirname = ["nq"]
#beirname = ["hotpotqa"]
#beirname = ["climate-fever"]
#beirname = ["dbpedia-entity"]#,"webis-touche2020","hotpotqa"]
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
# Load model directly
# print(args.model)

tokenizer = AutoTokenizer.from_pretrained(args.model)
#model = AutoModelForMaskedLM.from_pretrained("naver/efficient-splade-VI-BT-large-doc").to(device)
model = BertForMaskedLM.from_pretrained(args.model).to(device)
model.eval()
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

scale = args.scale
#file_per = 100000
i = 0


fo = None
#skip msmarco
for dsname in beirname:
    total_len = 0
    count = 0
    batchsz = 64 # batchsz has to be able to divide file_per
    dataset = load_dataset(f"BeIR/{dsname}", "corpus")["corpus"]
    fo = open(os.path.join(out_dir, f"{dsname}_file.jsonl"), "w+")
    #fmap = open(os.path.join(out_dir, f"{dsname}_idmapping.txt"),"w+")
    overflows = 0
    for i in trange(len(dataset)//batchsz+1,desc=f"{dsname}"):
        with torch.inference_mode():
            dids = dataset[i*batchsz:i*batchsz+batchsz]['_id']
            doc_title = dataset[i*batchsz:i*batchsz+batchsz]['title']
            doc_text = dataset[i*batchsz:i*batchsz+batchsz]['text']
            docs = [f"{title} {text}" for title,text in zip(doc_title,doc_text)]
            tokens = tokenizer(docs, return_tensors='pt', padding=True, truncation=True,max_length=512).to(device)
            doc_reps = model(**tokens)["logits"]#.squeeze()  # (sparse) doc rep in voc space, shape (30522,)
            #print(torch.max(torch.log(1 + torch.relu(doc_reps)) * tokens.attention_mask.unsqueeze(-1), dim=1).values.size())
            doc_reps = torch.max(torch.log(1 + torch.relu(doc_reps)) * tokens.attention_mask.unsqueeze(-1), dim=1).values
        # get the number of non-zero dimensions in the rep:c
        for idx, item in enumerate(zip(dids,doc_reps)):
            did, doc_rep = item
            doc_rep = doc_rep.cpu().numpy()
            doc_rep = np.clip(np.rint(doc_rep * scale).astype(int),0,255)
            col = np.nonzero(doc_rep)
            #print(col)
            #col = torch.nonzero(doc_rep).squeeze(-1).detach().cpu().tolist()
            weights = doc_rep[col]#.cpu().tolist()
            d = {reverse_voc[k]: int(v) for k, v in zip(col[0], weights)}
            #print(did)
            outline = json.dumps({"id": did, "content": doc_text[idx], "vector": d}) + "\n"
            fo.write(outline)
            count += 1
            total_len += len(col[0])
    print(f"{dsname} average document length {total_len/count}, overflow {overflows/count}")
    fo.close()
