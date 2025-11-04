import torch
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM
from tqdm import tqdm, trange
import gzip
import json
import os
import argparse
import numpy as np
import sys
import gc
MSMARCO_DOC_SZ = 8841823

parser = argparse.ArgumentParser()
agg = "max"
parser.add_argument('--model',type=str,help="model path")
parser.add_argument('--checkpoint',type=str,help="checkpoint path if needed, no necessary",default="")
parser.add_argument('--output',type=str,help="output prefix")
parser.add_argument('--id',default=0,type=int,help="only used when launching task in parallel")
parser.add_argument('--scale',type=int,help='score scale of the weight',default=100)
args = parser.parse_args()

model_type_or_dir = args.model

out_dir = args.output

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# loading model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForMaskedLM.from_pretrained(args.model).to(device)
if args.checkpoint != "":
    model.load_state_dict(torch.load(args.checkpoint))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

scale = args.scale
file_per = 2500000
i = 0


fo = gzip.open(os.path.join(out_dir, f"file_{args.id}.jsonl.gz"), "w")

corpus = []
with open("/projects/bcgk/zwang48/sclr/msmarco-full/collection.tsv") as f:
    for line in tqdm(f,total=MSMARCO_DOC_SZ):
        did, doc = line.split('\t')
        corpus.append(doc)
batchsz = 128 # batchsz has to be able to divide file_per
corpus = corpus[args.id*file_per:args.id*file_per+file_per]
total_len = 0
count = 0
for i in trange(len(corpus)//batchsz+1,desc=f"{args.id*file_per}->{args.id*file_per+file_per}"):
    with torch.no_grad():
        dids = range(i*batchsz,i*batchsz+batchsz)
        #doc_title = dataset[i*batchsz:i*batchsz+batchsz]['title']
        doc_text = corpus[i*batchsz:i*batchsz+batchsz]
        tokens = tokenizer(doc_text, return_tensors='pt', padding=True, truncation=True,max_length=512).to(device)
        doc_reps = model(**tokens)["logits"]#.squeeze()  # (sparse) doc rep in voc space, shape (30522,)
        #print(torch.max(torch.log(1 + torch.relu(doc_reps)) * tokens.attention_mask.unsqueeze(-1), dim=1).values.size())
        doc_reps = torch.max(torch.log(1 + torch.relu(doc_reps)) * tokens.attention_mask.unsqueeze(-1), dim=1).values
    # get the number of non-zero dimensions in the rep:c
    for idx, item in enumerate(zip(dids,doc_reps)):
        did, doc_rep = item
        col = torch.nonzero(doc_rep).squeeze(-1).detach().cpu().tolist()
        total_len += len(col)
        count += 1
        weights = doc_rep[col].cpu().tolist()
        d = {reverse_voc[k]: int(v * scale) for k, v in zip(col, weights)}
        outline = json.dumps({"id": args.id*file_per + did, "content": doc_text[idx], "vector": d}) + "\n"
        fo.write(outline.encode('utf-8'))
print(f"msmarco average document length {total_len/count}")
fo.close()