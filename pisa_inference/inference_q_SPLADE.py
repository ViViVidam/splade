import torch
from transformers import BertForMaskedLM, AutoTokenizer
import sys
from tqdm import tqdm
import os
import argparse
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
def flops(batch_rep):
    return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)
def entropy_loss(rep):
    ori_rep = rep
    flop = flops(rep)
    rep = torch.softmax(rep,dim=-1)
    #print(rep)
    rep = torch.sum(rep * ori_rep,dim=-1)
    return torch.mean(rep),flop
# loading model and tokenizer
#model = SpladeAvgFusionPooling.from_pretrained(args.model)
#model = SpladeFusionPooling.from_pretrained(args.model)
#print(args.model)
model = BertForMaskedLM.from_pretrained(args.model)
if args.checkpoint != "":
    model.load_state_dict(torch.load(args.checkpoint))
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(args.model)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

scale = args.scale
i = 0
avg_len = 0
el = 0
fl = 0
dsq = open("/projects/bcgk/yzound/datasets/msmarco/queries.small.dev.tsv","r").readlines()
with open(os.path.join(args.output, "queries.dev.tsv"), "w") as fo:
    for line in tqdm(dsq):
        did, doc = line.split('\t')
        with torch.no_grad():
            token = tokenizer(doc, return_tensors="pt", truncation=True).to(device)
            doc_rep = model(**token)["logits"]#.squeeze()  # (sparse) doc rep in voc space, shape (30522,)
            doc_rep = torch.max(torch.log(1 + torch.relu(doc_rep)) * token.attention_mask.unsqueeze(-1), dim=1).values.squeeze()
            s = entropy_loss(doc_rep)
            el += s[0].cpu()
            fl += s[1].cpu()
        # get the number of non-zero dimensions in the rep:
        doc_rep = torch.round(doc_rep*scale).clip(0,255)
        col = torch.nonzero(doc_rep).squeeze(-1).cpu().tolist()
        avg_len += len(col)
        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {reverse_voc[k]: int(v) for k, v in zip(col, weights)}
        q = []
        for tok in d:
            for _ in range(d[tok]):
                q.append(tok)
        #outline = json.dumps({"id": int(did), "content": doc, "vector": d}) + "\n"
        fo.write(f"{did}\t{' '.join(q)}\n")
        fo.flush()
        i += 1
print(f"avg len query: {avg_len/len(dsq)}")
print(f"avg el query: {el/len(dsq)}, avg fl query: {fl/len(dsq)}")