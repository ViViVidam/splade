from transformers import AutoModel,AutoTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer
from transformer_model.models import SpladeAvgFusionPooling, Splade, SpladeFusionPoolingUntie
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
def train(documentModel:str,queryModel:str,save_path:str,batchsz:int=64,accum_steps:int =4,save_steps:int=3000,global_steps:int=0,warmup_steps:int = 400,lambda_q:int = 0.01,lambda_d:int = 0.008,epochs:int = 5,use_untie_model:bool = False):
    ce_scores = getMSMARCOCEscore()
    corpus = getMSMARCOCorpus()
    queries = getMSMARCOQuery()
    training_samples = getSpladeHN(queries_texts=queries)
    corpus_embds = getLLamaCorpusEmbds()
    config = SpladeFuseMaxPConfig().from_pretrained(documentModel)
    if use_untie_model:
        modelDocument = SpladeFusionPoolingUntie(config=config,path=documentModel).to(device)
        qConfig = SpladeFuseMaxPConfig().from_pretrained("/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/0_MLMTransformer/")
        modelQuery = SpladeFusionPoolingUntie(config=qConfig).to(device)
    else:
        modelDocument = SpladeAvgFusionPooling(config=config,path=documentModel).to(device)
        modelQuery = SpladeAvgFusionPooling(config=config,path=queryModel).to(device)
    tokenizer = AutoTokenizer.from_pretrained("/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/0_MLMTransformer/")
    dataset = MSMARCODatasetExtraEmbedding(corpus=corpus,training_samples=training_samples,ce_scores=ce_scores,embeddings=corpus_embds,loss_type="marginmse",model_type="splade")
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batchsz//accum_steps, drop_last=True,collate_fn=collate_extraEmbds)
    del corpus_embds
    optimizerq = torch.optim.AdamW(modelQuery.parameters(),lr=3e-5)
    optimizerd = torch.optim.AdamW(modelDocument.parameters(),lr=3e-5)
    schedulerd = transformers.get_linear_schedule_with_warmup(optimizerd, num_warmup_steps=warmup_steps, num_training_steps=len(dataset)*epochs)
    schedulerq = transformers.get_linear_schedule_with_warmup(optimizerq, num_warmup_steps=warmup_steps, num_training_steps=len(dataset)*epochs)
    mseloss = torch.nn.MSELoss()
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
            fusedEmdsPos = torch.from_numpy(np.array([item['pos'] for item in affiliate])).to(device)
            fusedEmdsNeg = torch.from_numpy(np.array([item['neg'] for item in affiliate])).to(device)
            labels = torch.tensor(labels,device=device)
            query_tokens = tokenizer(queries,padding=True,truncation='longest_first',return_tensors='pt').to(device)
            doc_pos_tokens = tokenizer(doc_pos,padding=True,truncation='longest_first',return_tensors='pt').to(device)
            doc_neg_tokens = tokenizer(doc_neg,padding=True,truncation='longest_first',return_tensors='pt').to(device)
            doc_pos_tokens['inputs_dense_embeds'] = fusedEmdsPos
            doc_neg_tokens['inputs_dense_embeds'] = fusedEmdsNeg
            with torch.autocast(device_type=device):
                q = modelQuery(**query_tokens)
                pos_feature = modelDocument(**doc_pos_tokens)
                neg_feature = modelDocument(**doc_neg_tokens)
                pos_score = dot(q,pos_feature)
                neg_score = dot(q,neg_feature)
                margin_pred = pos_score - neg_score
                #print(margin_pred.size())
                flops_doc = lambda_d * flops(torch.cat([pos_feature,neg_feature], 0))
                flops_query = lambda_q * l1loss(q)
                #print(flops_query,flops_doc)
                loss = mseloss(margin_pred,labels) + flops_doc + flops_query
            scaler.scale(loss/accum_steps).backward()
            steps += 1
            loss_accum += loss
            if steps % accum_steps == 0:
                scaler.step(optimizerq)
                scaler.step(optimizerd)
                scaler.update()
                schedulerd.step()
                schedulerq.step()
                optimizerq.zero_grad()
                optimizerd.zero_grad()
            pbar.set_postfix_str(f"accumulated loss: {loss_accum/steps}, loss: {loss}")
            if (steps / accum_steps) % save_steps == 0:
                path = os.path.join(save_path,f"{steps//accum_steps}")
                if not os.path.exists(path):
                    os.mkdir(path)
                modelDocument.save_pretrained(os.path.join(path,"docModel"))
                modelQuery.save_pretrained(os.path.join(path,"queryModel"))
                torch.save(modelQuery.state_dict(), os.path.join(path,"model_q_state_dict.pt"))
                torch.save(modelDocument.state_dict(), os.path.join(path,"model_d_state_dict.pt"))
    modelDocument.save_pretrained(os.path.join(save_path,"docModel"))
    modelQuery.save_pretrained(os.path.join(save_path,"queryModel"))
    torch.save(modelQuery.state_dict(), os.path.join(save_path,"model_q_state_dict.pt"))
    torch.save(modelDocument.state_dict(), os.path.join(save_path,"model_d_state_dict.pt"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_untie',type=bool,default=False)
    parser.add_argument('--save_path',type=str,default=f"/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model")
    parser.add_argument('--docModel',type=str,default="/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/0_MLMTransformer/")
    parser.add_argument('--queryModel',type=str,default="/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/0_MLMTransformer/")
    parser.add_argument('--steps',type=int,default=0)
    args = parser.parse_args()
    train(documentModel=args.docModel,queryModel=args.queryModel,global_steps=args.steps,save_path=args.save_path,use_untie_model=args.use_untie)