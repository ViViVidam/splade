from transformers import AutoTokenizer, BertForMaskedLM
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from data.data import *
from torch.utils.data import DataLoader
from transformer_model.models import AutoModelForMaskedLMUntie
import torch
import argparse
import tqdm
from loss import *
import transformers
import os
from accelerate import Accelerator, load_checkpoint_and_dispatch
from accelerate.utils import DistributedDataParallelKwargs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_untie',type=bool,default=False)
    parser.add_argument('--checkpoint',type=str,default="")
    parser.add_argument('--save_path',type=str,required=True, default=f"/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model")
    parser.add_argument('--docModel',type=str,default="/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/0_MLMTransformer/")
    parser.add_argument('--queryModel',type=str,default="/expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/0_MLMTransformer/")
    parser.add_argument('--steps',type=int,default=0)
    parser.add_argument('--epoch',type=int,default=5)
    parser.add_argument('--gradient_accum',type=int,default=4)
    parser.add_argument('--batchsize',type=int,default=128)
    args = parser.parse_args()
    return args

def train(accelerator, epochs:int,checkpoint:str,documentModel:str,save_path:str,batchsz:int=32,save_steps:int=20000,global_steps:int=0,warmup_steps:int = 2000,lambda_q:int = 0.01,lambda_d:int = 0.008,use_untie:bool = False):
    device = accelerator.device
    ce_scores = getMSMARCOCEscore()
    num_of_neg = 5
    corpus = getMSMARCOCorpus()
    queries = getMSMARCOQuery()
    #training_samples = getTevaronSamples()
    training_samples = load_dataset("sentence-transformers/msmarco-bm25","triplet-50-ids")["train"]
    '''
    device_map = {
    "bert.embeddings": 1,
    "bert.encoder":0,
    "bert.pooler":0,
    "cls": 1,
    }
    '''
    mini_batchsz = batchsz 
    model = BertForMaskedLM.from_pretrained(documentModel)
    tokenizer = AutoTokenizer.from_pretrained(documentModel)
    if use_untie:
        print("*******************\nuntie the model embeddings..\n*******************",flush=True)
        untied_embds = torch.nn.Embedding(len(tokenizer.vocab),768)
        untied_embds.weight.data = model.get_input_embeddings().weight.clone()
        model.set_input_embeddings(untied_embds)
    if len(checkpoint) > 0:
        model.load_state_dict(torch.load(os.path.join(checkpoint),weights_only=True))
    model.to(device)
    dataset = MSMARCODataset(dataset=training_samples,queries=queries,corpus=corpus,ce_scores=ce_scores,num_of_neg=num_of_neg)
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=mini_batchsz, drop_last=True)
    optimizer = transformers.AdamW(model.parameters(),lr=3e-5)#torch.optim.AdamW(model.parameters(),lr=4e-6)
    # step不能太高 3e-4会直接不动
    total_steps = len(dataset) // batchsz
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps*epochs)
    flopsSchedulerD = RegWeightScheduler(lambda_=2e-2,T=5*total_steps//accelerator.num_processes) # after finish 10% epoch, the lambda will reach to 2e-2 epochs*total_len//10
    flopsSchedulerQ = RegWeightScheduler(lambda_=2e-2,T=5*total_steps//accelerator.num_processes)
    #loss_fct= torch.nn.CrossEntropyLoss()
    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
    #loss_fct = torch.nn.MSELoss()
    flops = FLOPS()
    steps = global_steps
    train_dataloader,model,optimizer= accelerator.prepare(train_dataloader,model,optimizer)
    scheduler = accelerator.prepare_scheduler(scheduler)
    for epoch in range(epochs):
        for batch in (pbar:=tqdm.tqdm(train_dataloader,desc=f"epoch {epoch}", disable=not accelerator.is_local_main_process)):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                queries = batch['query']
                passages = batch['passage']
                scores = batch['ce_scores'].to(device)
                #labels = torch.zeros(batchsz//accum_steps,dtype=torch.long,device=device)
                query_tokens = tokenizer(queries,padding=True,truncation='longest_first',return_tensors='pt').to(device)
                qfeature = model(**query_tokens)["logits"]
                qfeature = torch.max(torch.log(1 + torch.relu(qfeature)) * query_tokens.attention_mask.unsqueeze(-1), dim=1).values
                '''
                doc_pos_tokens = tokenizer(doc_pos,padding=True,truncation='longest_first',return_tensors='pt').to(device)
                pos_feature = model(**doc_pos_tokens)["logits"]
                pos_feature = torch.max(torch.log(1 + torch.relu(pos_feature)) * doc_pos_tokens.attention_mask.unsqueeze(-1), dim=1).values
                pos_score = torch.matmul(q,torch.transpose(pos_feature,0,1))
                doc_neg_tokens = tokenizer(doc_neg,padding=True,truncation='longest_first',return_tensors='pt').to(device)
                neg_feature = model(**doc_neg_tokens)["logits"]
                neg_feature = torch.max(torch.log(1 + torch.relu(neg_feature)) * doc_neg_tokens.attention_mask.unsqueeze(-1), dim=1).values
                neg_score = dot(q,neg_feature)
                '''
                predict = []
                flops_doc = 0
                #print(len(passages))
                for i in range(len(passages)):
                    input = passages[i]
                    tokens = tokenizer(input,padding=True,truncation='longest_first',return_tensors='pt').to(device)
                    feature = model(**tokens)["logits"]
                    feature = torch.max(torch.log(1 + torch.relu(feature)) * tokens.attention_mask.unsqueeze(-1), dim=1).values
                    predict.append(dot(qfeature,feature).unsqueeze(-1))
                    flops_doc += flopsSchedulerD.get_lambda() * flops(feature)
                #print(predict)
                predict = torch.concat(predict,dim=-1)
                #print(predict)
                scores = batch['ce_scores'].to(device)
                
                predict = torch.nn.functional.log_softmax(predict,dim=-1)
                scores = torch.nn.functional.softmax(scores,dim=-1)
                
                #loss = loss_fct(predict[:,0]-predict[:,1],scores[:,0]-scores[:,1])
                loss = loss_fct(predict,scores)
                flops_doc = flops_doc / (num_of_neg + 1)
                flops_query = flopsSchedulerQ.get_lambda() * flops(qfeature)
                loss = loss + flops_doc + flops_query
                #(loss/accum_steps).backward()
                accelerator.backward(loss)
                steps += 1
                optimizer.step()
                scheduler.step()
                flopsSchedulerQ.step()
                flopsSchedulerD.step()
                if steps % 100 == 0:
                    pbar.set_postfix_str(f"loss: {loss.item()}, d_flops: {flopsSchedulerD.get_lambda()}")
                if steps % save_steps == 0 and accelerator.is_main_process:
                    #accelerator.wait_for_everyone()
                    path = os.path.join(save_path,f"{steps}")
                    if not os.path.exists(path):
                        os.mkdir(path)
                    accelerator.unwrap_model(model).save_pretrained(os.path.join(path,"spladeOrignal"),is_main_process=accelerator.is_main_process,save_function=accelerator.save)
    #accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(os.path.join(save_path,"spladeOrignal"),is_main_process=accelerator.is_main_process,save_function=accelerator.save)
    print("model save to {}".format(os.path.join(save_path,"model_state_dict.pt")))

#torch DDP
def run(rank, world_size,args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    train(batchsz=args.batchsize,epochs=args.epoch,checkpoint=args.checkpoint, documentModel=args.docModel,queryModel=args.queryModel,global_steps=args.steps,save_path=args.save_path,use_untie_model=args.use_untie)
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    args = get_args()
    accelerator = Accelerator(mixed_precision="fp16",gradient_accumulation_steps=args.gradient_accum,kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)])
    train(accelerator= accelerator, batchsz=args.batchsize//args.gradient_accum,epochs=args.epoch,checkpoint=args.checkpoint, documentModel=args.docModel,global_steps=args.steps,save_path=args.save_path,use_untie=args.use_untie)
    #n_gpus = torch.cuda.device_count()
    #world_size = n_gpus//2
    #mp.spawn(run,args=(world_size,args),nprocs=world_size,join=True)