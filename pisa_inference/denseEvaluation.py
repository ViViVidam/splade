import faiss
from transformers import BertTokenizer
from datasets import load_dataset
import torch
import numpy as np
from transformer_model import *
from typing import DefaultDict
import tqdm
import pickle
import fnmatch
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,help="model path")
parser.add_argument('--threshold',type=int,default=0,help="indexing threshold")
parser.add_argument('--output',type=str,help="output prefix")
parser.add_argument('--id',default=0,type=int,help="only used when launching task in parallel")
parser.add_argument('--scale',type=int,help='score scale of the weight',default=100)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"

#### Provide the data_path where scifact has been downloaded and unzipped
qembds,qids = pickle.load(open(f"/expanse/lustre/projects/csb176/she2/tevatron/examples/repllama/beir_embedding_scifact/queries_{dataset}.0.pkl","rb"))
repllama_embds = DefaultDict()
filenames = fnmatch.filter(os.listdir(f"/expanse/lustre/projects/csb176/she2/tevatron/examples/repllama/beir_embedding_{dataset}/"),'corpus_*')
for filename in filenames:
    corpus_embds_file = os.path.join(f"/expanse/lustre/projects/csb176/she2/tevatron/examples/repllama/beir_embedding_{dataset}/",filename)
    dembds, dids = pickle.load(open(corpus_embds_file,"rb"))
    for dembd, did in zip(dembds,dids):
        repllama_embds[did] = dembd
corpus = load_dataset(f"BeIR/{dataset}", "corpus")["corpus"]
corpus_emds = []
#corpus {text: , title:}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocabsz = len(tokenizer.get_vocab())
model = SpladeFusionPooling.from_pretrained(args.model)
#model = SpladeAvgFusionPooling.from_pretrained(args.model)
model.load_state_dict(torch.load("/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model/tie/12000/model_d_state_dict.pt"))
model.eval()
model.to(device)
idx2docid = []
for i,item in tqdm.tqdm(enumerate(corpus),total=len(corpus)):
    with torch.no_grad():
        docid = item["_id"] # class(doc_id) = str
        doc = item["text"]
        if len(item["title"]) > 0:
            doc = item["title"] + " " + doc
        with torch.no_grad():
            embds = torch.from_numpy(repllama_embds[docid]).to(device)
            token = tokenizer(doc, return_tensors="pt", truncation=True).to(device)
            if(len(embds.size()) == 1):
                embds = embds.unsqueeze(0)
            token['inputs_dense_embeds'] = embds
            output = model.inference(**token).squeeze()
        corpus_emds.append(output.detach().cpu().numpy())
        idx2docid.append(docid)
corpus_emds = np.array(corpus_emds)
d = np.shape(corpus_emds)[1]
print(d)
index = faiss.IndexFlatIP(d)
index.add(corpus_emds)
#device = faiss.StandardGpuResources()
#index = faiss.index_cpu_to_gpu(device, 0, index)
k = 1000

dsq = load_dataset(f"BeIR/scifact", "queries")["queries"]
queries_emds = []
modelq = SpladeFusionPooling.from_pretrained(args.model).to(device)
modelq.load_state_dict(torch.load("/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model/tie/12000/model_q_state_dict.pt"))
for item in tqdm.tqdm(dsq):
    did, doc = item["_id"], item["text"]
    with torch.no_grad():
        token = tokenizer(doc, return_tensors="pt", truncation=True).to(device)
        output = modelq(**token).squeeze()
        queries_emds.append(output.detach().cpu().numpy())
queries_emds = np.array(queries_emds)
distances,neighbors = index.search(queries_emds, k)
fout = open("out.trec","w+")
for i,item in tqdm.tqdm(enumerate(dsq),total=len(dsq)):
    rank = 1
    qid, doc = item["_id"], item["text"]
    for doc_id,dist in zip(neighbors[i],distances[i]):
        '''
        qid, Q0, did, rank_in_query, score, runid
        msmarco trec submission format
        '''
        fout.write(f"{qid}\tQ0\t{idx2docid[doc_id]}\t{rank}\t{dist}\t0\n")#doc_id
        rank += 1
fout.close()