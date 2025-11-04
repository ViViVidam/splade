import argparse
from transformers import AutoModelForMaskedLM, BertTokenizerFast
import datasets
from tqdm import tqdm
import torch
device = 'cuda'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default="sentence-transformers/msmarco-bert-co-condensor")
    parser.add_argument('--threshold',type=float,default=0)
    args = parser.parse_args()
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    tokenizer = BertTokenizerFast.from_pretrained(args.model)
    model.to(device)
    model.eval()
    ds = datasets.load_dataset("Tevatron/msmarco-passage-corpus","default")["train"]
    batchsize = 64
    total_length = len(ds) // batchsize
    avg_len = 0
    if len(ds)%batchsize:
        total_length += 1
    with torch.no_grad():
        for i in tqdm(range(total_length)):
            items = ds[i*batchsize:min(len(ds),i*batchsize+batchsize)]
            tokens = tokenizer(items["text"],return_tensors='pt',padding=True,truncation=True).to(device)
            doc_reps = model(**tokens)["logits"]
            doc_reps = torch.max(torch.log(1 + torch.relu(doc_reps-args.threshold)) * tokens.attention_mask.unsqueeze(-1), dim=1).values
            #print(torch.nonzero(doc_reps).shape)
            avg_len += (torch.count_nonzero(doc_reps,dim=-1)/len(ds)).sum().item()
    
    print(f"avg length of {args.model}, threshold {args.threshold} : {avg_len}")