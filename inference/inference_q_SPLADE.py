import torch
from transformers import BertForMaskedLM, AutoTokenizer
import sys
from tqdm import tqdm
from datasets import load_dataset
import os
import argparse
from transformer_model import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
agg = "max"
parser.add_argument('--model',type=str,help="model path")
parser.add_argument('--checkpoint',type=str,help="checkpoint path if needed, no necessary",default="")
parser.add_argument('--output',type=str,help="output prefix")
parser.add_argument('--scale',type=int,help='score scale of the weight',default=50)
parser.add_argument('--dataset',type=str,default="scifact")

args = parser.parse_args()
model_type_or_dir = args.model
out_dir = args.output

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# loading model and tokenizer
#model = SpladeAvgFusionPooling.from_pretrained(args.model)
#model = SpladeFusionPooling.from_pretrained(args.model)
model = BertForMaskedLM.from_pretrained(args.model)
if args.checkpoint != "":
    model.load_state_dict(torch.load(args.checkpoint))
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/0_MLMTransformer/")
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

scale = args.scale
i = 0

dsq = load_dataset(f"BeIR/{args.dataset}", "queries")["queries"]
with open(os.path.join(args.output, "queries.train.tsv"), "w") as fo:
    for line in tqdm(dsq):
        did, doc = line["_id"], line["text"]
        with torch.no_grad():
            token = tokenizer(doc, return_tensors="pt", truncation=True).to(device)
            doc_rep = model(**token)["logits"]#.squeeze()  # (sparse) doc rep in voc space, shape (30522,)
            doc_rep = torch.max(torch.log(1 + torch.relu(doc_rep)) * token.attention_mask.unsqueeze(-1), dim=1).values.squeeze()
        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {reverse_voc[k]: int(v * scale) for k, v in zip(col, weights)}
        q = []
        for tok in d:
            for _ in range(d[tok]):
                q.append(tok)
        #outline = json.dumps({"id": int(did), "content": doc, "vector": d}) + "\n"
        fo.write(f"{did}\t{' '.join(q)}\n")
        fo.flush()
        i += 1
