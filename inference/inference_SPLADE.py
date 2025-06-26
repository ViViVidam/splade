import torch
from transformers import AutoTokenizer, BertForMaskedLM
from tqdm import tqdm, trange
import gzip 
import json
import os
import argparse
import numpy as np
from datasets import load_dataset
from transformer_model.models import *
from transformer_model.modelConfig import SpladeFuseMaxPConfig
import pickle
from typing import DefaultDict
parser = argparse.ArgumentParser()
agg = "max"
parser.add_argument('--model',type=str,help="model path")
parser.add_argument('--checkpoint',type=str,help="checkpoint path if needed, no necessary",default="")
parser.add_argument('--dataset',type=str,default="scifact")
parser.add_argument('--output',type=str,help="output prefix")
parser.add_argument('--id',default=0,type=int,help="only used when launching task in parallel")
parser.add_argument('--scale',type=int,help='score scale of the weight',default=100)
parser.add_argument('--use_llama',type=bool,default = False)
args = parser.parse_args()

model_type_or_dir = args.model

out_dir = args.output

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# loading model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#config = SpladeFuseMaxPConfig().from_pretrained(args.model)
#model = SpladeFusionPooling.from_pretrained(args.model)
model = BertForMaskedLM.from_pretrained(args.model)
#model = SpladeFusionPoolingUntie.from_pretrained(args.model)
#model = SpladeAvgFusionPooling.from_pretrained(args.model)
if args.checkpoint != "":
    model.load_state_dict(torch.load(args.checkpoint))
model.eval()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/0_MLMTransformer/")
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

scale = args.scale
file_per = 100000
i = 0

dataset_name = args.dataset
ds = load_dataset(f"BeIR/{dataset_name}", "corpus")["corpus"]
embeddings = DefaultDict()
for i in range(4):
    dembds, dids = pickle.load(open(f"/expanse/lustre/projects/csb176/she2/tevatron/examples/repllama/beir_embedding_scifact/corpus_scifact.{i}.pkl","rb"))
    for embds,did in zip(dembds,dids):
        embeddings[int(did)] = embds

fo = gzip.open(os.path.join(out_dir, f"file_{args.id}.jsonl.gz"), "w")
for idx, line in tqdm(enumerate(ds),total=len(ds)):
    did = line["_id"]
    doc = line["text"]
    if len(line["title"]) > 0:
        doc = line["title"] + " " + doc
    embds = torch.from_numpy(embeddings[int(did)]).to(device)
    assert(len(embds)>0)
    with torch.no_grad():
        token = tokenizer(doc, return_tensors="pt", truncation=True).to(device)
        if(len(embds.size()) == 1):
            embds = embds.unsqueeze(0)
        if args.use_llama:
            token['inputs_dense_embeds'] = embds
        doc_rep = model(**token)["logits"]#.squeeze()  # (sparse) doc rep in voc space, shape (30522,)
        doc_rep = torch.max(torch.log(1 + torch.relu(doc_rep)) * token.attention_mask.unsqueeze(-1), dim=1).values.squeeze()
    # get the number of non-zero dimensions in the rep:
    col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
    #print("number of actual dimensions: ", len(col))
    # now let's inspect the bow representation:
    weights = doc_rep[col].cpu().tolist()
    d = {reverse_voc[k]: int(scale * v) for k, v in zip(col, weights) if v*scale >= 1}
    outline = json.dumps({"id": int(did), "content": doc, "vector": d}) + "\n"
    fo.write(outline.encode('utf-8'))
    fo.flush()
    i += 1

fo.close()