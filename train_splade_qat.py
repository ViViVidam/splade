from transformers import AutoTokenizer, BertForMaskedLM, AutoConfig
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from data.data import *
from torch.utils.data import DataLoader
from accelerate import Accelerator, load_checkpoint_and_dispatch
from datasets import load_dataset
import torch
import argparse
import tqdm
from loss import *
import transformers
import os
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig


from accelerate.utils import DistributedDataParallelKwargs, FullyShardedDataParallelPlugin

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(100*input).clip(0,255) / 100
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output #torch.nn.functional.hardtanh(grad_output)
class StraightThroughEstimator(torch.nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()
    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class Wrapper(torch.nn.Module):
    def __init__(self,model):
        super(Wrapper,self).__init__()
        self.qat = StraightThroughEstimator()
        self.model = model
        self.config = model.config
        
    def save_pretrained(self,*args,**kwargs):
        return self.model.save_pretrained(*args,**kwargs)
    
    def forward(self,*args,**kwargs):
        #print(args)
        #print(kwargs)
        feat = self.model(**kwargs)["logits"]
        #print(feat)
        #print(kwargs.get('attention_mask'))
        feat = torch.log(torch.relu(torch.max(feat + ( 1 - kwargs.get('attention_mask').unsqueeze(-1)) * -1e6, dim=1)[0]) + 1)
        nonz = torch.nonzero(feat,as_tuple=True)
        #print(feat[nonz])
        feat_qat = self.qat(feat)
        #print(feat_qat[nonz])
        return feat,feat

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
    parser.add_argument('--per_device_batch_size',type=int,default=64) #per device batch size
    args = parser.parse_args()
    return args

def train(accelerator, epochs:int,checkpoint:str,documentModel:str,save_path:str,batchsz:int=32,save_steps:int=20000,global_steps:int=0,warmup_steps:int = 4000,lambda_q:int = 0.05,lambda_d:int = 0.02,use_untie:bool = False):
    device = accelerator.device
    scheduler_ratio = 0.001
    ce_scores = getMSMARCOCEscore(path="/projects/bcgk/yzound/datasets/msmarco/msmarco_train_teacher_scores.jsonl")
    num_of_neg = 5
    corpus = getMSMARCOCorpus("/projects/bcgk/zwang48/sclr/msmarco-full/collection.tsv")
    #config = AutoConfig.from_pretrained(documentModel)
    #config.tie_word_embeddings = False
    
    per_device_batch_size = batchsz // args.gradient_accum
    model = BertForMaskedLM.from_pretrained(documentModel)#,config=config)
    model = Wrapper(model=model)
    #model("")
    #accelerator.unwrap_model(model).save_pretrained("tmp/",is_main_process=accelerator.is_main_process,save_function=accelerator.save,state_dict=accelerator.get_state_dict(model.model))
    #BertForMaskedLM.from_pretrained("tmp")
    #model.tie_weights()
    tokenizer = AutoTokenizer.from_pretrained(documentModel)
    tokenizer.save_pretrained(save_path)
    if use_untie:
        print("*******************\nuntie the model embeddings..\n*******************",flush=True)
        untied_embds = torch.nn.Embedding(len(tokenizer.vocab),768)
        untied_embds.weight.data = model.get_input_embeddings().weight.clone()
        model.set_input_embeddings(untied_embds)
    if len(checkpoint) > 0:
        model.load_state_dict(torch.load(os.path.join(checkpoint),weights_only=True))
    model.to(device)
    dataset = MSMARCODataset(corpus=corpus,ce_scores=ce_scores,num_of_neg=num_of_neg)
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=per_device_batch_size, drop_last=True, num_workers=accelerator.num_processes)
    lr = 2e-5
    #print(per_device_batch_size)
    #print(len(train_dataloader))
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)#torch.optim.AdamW(model.parameters(),lr=4e-6)
    # step不能太高 3e-4会直接不动
    total_steps = len(dataset) // batchsz
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*epochs*0.01), num_training_steps=total_steps*epochs)
    accelerator.init_trackers(
        #name="spladeV3-3e-5-hn-8",
        project_name="splade",
        config={
            'model_config': model.config,
            'lr': lr,
            'batchsize':per_device_batch_size * args.gradient_accum * accelerator.state.num_processes,
            'epoch': epochs,
            'location': args.save_path,
            'lambda':[lambda_q,lambda_d],
            'scheduler_ratio':scheduler_ratio,
        }
        , init_kwargs={"wandb":{"name":"spladeV3-3e-5-hn-8-qat"}}
    )
    #loss_fct= torch.nn.CrossEntropyLoss()
    loss_fct1 = torch.nn.KLDivLoss(reduction="batchmean")
    loss_fct2 = torch.nn.MSELoss()
    flops = FLOPS()
    steps = global_steps
    train_dataloader,model,optimizer= accelerator.prepare(train_dataloader,model,optimizer)
    # print(len(train_dataloader))
    scheduler = accelerator.prepare_scheduler(scheduler)
    flopsSchedulerD = RegWeightScheduler(lambda_=lambda_d,T=scheduler_ratio*epochs*len(train_dataloader)) # after finish 10% epoch, the lambda will reach to max epochs*total_len//10
    flopsSchedulerQ = RegWeightScheduler(lambda_=lambda_q,T=scheduler_ratio*epochs*len(train_dataloader))
    for epoch in range(epochs):
        for batch in (pbar:=tqdm.tqdm(train_dataloader,desc=f"epoch {epoch}", disable=not accelerator.is_local_main_process)):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                queries = batch['query']
                passages = batch['passage']
                scores = batch['ce_scores']
                #labels = torch.zeros(batchsz//accum_steps,dtype=torch.long,device=device)
                query_tokens = tokenizer(queries,padding=True,truncation='longest_first',return_tensors='pt',max_length=256).to(device)
                qfeature_qat,qfeature = model(**query_tokens)
                #qfeature = torch.log(torch.relu(torch.max(qfeature + ( 1 - query_tokens.attention_mask.unsqueeze(-1)) * -1e6, dim=1)[0]) + 1) #save memory
                #qfeature = torch.max(torch.log(1 + torch.relu(qfeature)) * query_tokens.attention_mask.unsqueeze(-1), dim=1).values
                predict = []
                flops_doc = 0
                #print(len(passages))
                for i in range(len(passages)):
                    input = passages[i]
                    tokens = tokenizer(input,padding=True,truncation='longest_first',return_tensors='pt',max_length=256).to(device)
                    feature_qat,feature = model(**tokens)
                    #feature = torch.log(torch.relu(torch.max(feature + ( 1 - tokens.attention_mask.unsqueeze(-1)) * -1e6, dim=1)[0]) + 1)
                    #feature = torch.max(torch.log(1 + torch.relu(feature)) * tokens.attention_mask.unsqueeze(-1), dim=1).values
                    predict.append(dot(qfeature_qat,feature_qat).unsqueeze(-1))
                    flops_doc += flopsSchedulerD.get_lambda() * flops(feature)
                #print(predict)
                predict = torch.concat(predict,dim=-1)
                #print(predict)
                scores = batch['ce_scores'].to(device)
                
                predict = torch.nn.functional.log_softmax(predict,dim=-1)
                scores = torch.nn.functional.softmax(scores,dim=-1)
                
                mse_loss = loss_fct2(predict[:,0]-predict[:,1],scores[:,0]-scores[:,1])
                kl_loss = loss_fct1(predict,scores)
                flops_doc = flops_doc / (num_of_neg + 1)
                flops_query = flopsSchedulerQ.get_lambda() * flops(qfeature)
                loss = kl_loss + 0.05 * mse_loss + flops_doc + flops_query
                accelerator.backward(loss)
                if steps % 100 == 0:
                    pbar.set_postfix_str(f"loss: {loss.item()}, d_flops: {flopsSchedulerD.get_lambda()}")
                    accelerator.log({"loss":loss.item(),'kl_loss':kl_loss.item(),'mse_loss':mse_loss.item(),'flops_doc':flops_doc.item(),'flops_query':flops_query.item(),'learning rate':optimizer.param_groups[0]['lr'],'flops_q_weight':flopsSchedulerQ.get_lambda(),'flops_d_weight':flopsSchedulerD.get_lambda()},step=steps)
                if steps !=0 and steps % save_steps == 0 and accelerator.is_main_process:
                    #accelerator.wait_for_everyone()
                    #print("saved")
                    out_index_dir = os.path.join(save_path,f"{steps}")
                    os.makedirs(out_index_dir, exist_ok=True)
                    accelerator.unwrap_model(model).save_pretrained(out_index_dir,is_main_process=accelerator.is_main_process,save_function=accelerator.save,state_dict=accelerator.get_state_dict(accelerator.unwrap_model(model).model))
                optimizer.step()
                scheduler.step()
                flopsSchedulerQ.step()
                flopsSchedulerD.step()
                steps += 1
    #accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(save_path,is_main_process=accelerator.is_main_process,save_function=accelerator.save,state_dict=accelerator.get_state_dict(accelerator.unwrap_model(model).model))
    accelerator.end_training()
    print("model save to {}".format(os.path.join(save_path,"model_state_dict.pt")))

if __name__ == "__main__":
    args = get_args()
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
    )
    accelerator = Accelerator(log_with="wandb",mixed_precision="fp16",gradient_accumulation_steps=args.gradient_accum,kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)])#,fsdp_plugin=fsdp_plugin)
    train(accelerator= accelerator, batchsz=args.per_device_batch_size,epochs=args.epoch,checkpoint=args.checkpoint, documentModel=args.docModel,global_steps=args.steps,save_path=args.save_path,use_untie=args.use_untie)
    #n_gpus = torch.cuda.device_count()
    #world_size = n_gpus//2
    #mp.spawn(run,args=(world_size,args),nprocs=world_size,join=True)