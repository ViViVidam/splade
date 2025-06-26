from transformers import AutoModel,AutoTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer
from transformer_model.models import SpladeFusionPooling, Splade, SpladeFusionPoolingUntie
from transformer_model.modelConfig import SpladeFuseMaxPConfig
from data.data import *
from torch.utils.data import DataLoader
import torch
import argparse
import tqdm
from loss import *
import transformers
import os
#import psutil

#process = psutil.Process(os.getpid())
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
def train(epochs:int,checkpoint:str,documentModel:str,queryModel:str,save_path:str,batchsz:int=64,accum_steps:int =4,save_steps:int=5000,global_steps:int=0,warmup_steps:int = 400,lambda_q:int = 0.01,lambda_d:int = 0.008,use_untie_model:bool = False):
    ce_scores = getMSMARCOCEscore()
    corpus = getMSMARCOCorpus()
    queries = getMSMARCOQuery()
    training_samples = getSpladeHN(queries_texts=queries)
    corpus_embds = getLLamaCorpusEmbds()
    config = SpladeFuseMaxPConfig.from_pretrained(documentModel)
    model = SpladeFusionPooling(config,queryModel).to(device)
    if len(checkpoint) > 0:
        model.load_state_dict(torch.load(os.path.join(checkpoint),weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained("/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/0_MLMTransformer/")
    num_neg = 3
    dataset = MSMARCODatasetExtraEmbedding(corpus=corpus,training_samples=training_samples,ce_scores=ce_scores,embeddings=corpus_embds,loss_type="kldiv",num_neg=num_neg,model_type="splade")
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batchsz//accum_steps, drop_last=True,collate_fn=collate_extraEmbds)
    del corpus_embds
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(dataset)*epochs)
    flopsScheduler = RegWeightScheduler(lambda_=2e-2,T=len(dataset)) # after finish one epoch, the lambda will reach to 2e-2
    klloss = torch.nn.KLDivLoss(reduction ="batchmean")
    #loss_fct= torch.nn.CrossEntropyLoss()
    l1loss = L1()
    if device == 'cpu':
        scaler = torch.GradScaler('cpu')
    else:
        scaler = torch.GradScaler('cuda')
    flops = FLOPS()
    steps = global_steps
    loss_accum = 0
    for epoch in range(epochs):
        for batch in (pbar:=tqdm.tqdm(train_dataloader,desc=f"epoch {epoch}")):
            queries,doc_pos,doc_neg, labels, affiliate = batch
            labels = torch.tensor(labels,device=device)
            #labels = torch.zeros(batchsz//accum_steps,dtype=torch.long,device=device)
            query_tokens = tokenizer(queries,padding=True,truncation='longest_first',return_tensors='pt').to(device)
            with torch.autocast(device_type=device):
                scores = []
                q = model(**query_tokens)
                doc_pos_tokens = tokenizer(doc_pos,padding=True,truncation='longest_first',return_tensors='pt').to(device)
                pos_feature = model(**doc_pos_tokens)
                scores.append(dot(q,pos_feature))
                for i in range(num_neg):
                    texts = [doc[i] for doc in doc_neg]
                    doc_neg_tokens = tokenizer(texts,padding=True,truncation='longest_first',return_tensors='pt').to(device)
                    neg_feature = model(**doc_neg_tokens)
                    neg_score = dot(q,neg_feature)
                    scores.append(neg_score)
                    #print(margin_pred.size())
                scores = torch.stack(scores).T
                scores = torch.nn.functional.log_softmax(scores,dim=-1)
                labels = torch.nn.functional.softmax(labels,dim=-1)
                loss = klloss(scores,labels)
                flops_doc = flopsScheduler.get_lambda() * flops(pos_feature) + flopsScheduler.get_lambda() * flops(neg_feature)
                flops_query = flopsScheduler.get_lambda() * flops(q)
                #print(flops_query,flops_doc)
                loss = loss + flops_doc + flops_query
            scaler.scale(loss/accum_steps).backward()
            steps += 1
            loss_accum += loss.detach()
            if steps % accum_steps == 0:
                #scaler.step(optimizerq)
                #scaler.step(optimizerd)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                #schedulerd.step()
                #schedulerq.step()
                flopsScheduler.step()
                #optimizerq.zero_grad()
                #optimizerd.zero_grad()
                optimizer.zero_grad()
            model.save_pretrained("tmp/spladeOrignal")
            pbar.set_postfix_str(f"accumulated loss: {loss_accum/steps}, loss: {loss.detach()}")
            if (steps / accum_steps) % save_steps == 0:
                path = os.path.join(save_path,f"{steps//accum_steps}")
                if not os.path.exists(path):
                    os.mkdir(path)
                #modelDocument.save_pretrained(os.path.join(path,"docModel"))
                #modelQuery.save_pretrained(os.path.join(path,"queryModel"))
                model.save_pretrained(os.path.join(path,"spladeOrignal"))
                #torch.save(modelQuery.state_dict(), os.path.join(path,"model_q_state_dict.pt"))
                #torch.save(modelDocument.state_dict(), os.path.join(path,"model_d_state_dict.pt"))
                torch.save(model.state_dict(), os.path.join(path,"model_state_dict.pt"))
                print(f"model save to {path}")
    model.save_pretrained(os.path.join(path,"spladeOrignal"))
    #modelDocument.save_pretrained(os.path.join(save_path,"docModel"))
    #modelQuery.save_pretrained(os.path.join(save_path,"queryModel"))
    torch.save(model.state_dict(), os.path.join(path,"model_state_dict.pt"))
    #torch.save(modelQuery.state_dict(), os.path.join(save_path,"model_q_state_dict.pt"))
    #torch.save(modelDocument.state_dict(), os.path.join(save_path,"model_d_state_dict.pt"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_untie',type=bool,default=False)
    parser.add_argument('--checkpoint',type=str,default="")
    parser.add_argument('--save_path',type=str,default=f"/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model")
    parser.add_argument('--docModel',type=str,default="/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/0_MLMTransformer/")
    parser.add_argument('--queryModel',type=str,default="/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/0_MLMTransformer/")
    parser.add_argument('--steps',type=int,default=0)
    parser.add_argument('--epoch',type=int,default=5)
    parser.add_argument('--batchsize',type=int,default=128)
    args = parser.parse_args()
    train(batchsz=args.batchsize,epochs=args.epoch,checkpoint=args.checkpoint, documentModel=args.docModel,queryModel=args.queryModel,global_steps=args.steps,save_path=args.save_path,use_untie_model=args.use_untie)