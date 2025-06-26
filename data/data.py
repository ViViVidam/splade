from torch.utils.data import Dataset
import random
import torch
import numpy as np
from typing import DefaultDict,Dict
import json
import gzip
import pickle
from typing import Tuple, List
from .data_utils import InputExample
from tqdm import tqdm
from datasets import load_dataset
# We create a custom MS MARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
ce_threshold = -3
MSMARCO_DOC_SZ = 8841823


def collate_extraEmbds(batch: List[InputExample]) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of InputExample instances: [InputExample(...), ...]

        Args:
            batch: a batch from a SmartBatchingDataset

        Returns:
            a batch of tensors for the model
        """
        queries = [example.texts[0] for example in batch]
        doc_pos = [example.texts[1] for example in batch]
        doc_neg = [example.texts[2:] for example in batch]
        labels = [example.label for example in batch]
        affiliate = [example.affiliate for example in batch]
        # Use torch.from_numpy to convert the numpy array directly to a tensor,
        # which is the recommended approach for converting numpy arrays to tensors


        return queries, doc_pos,doc_neg,labels, affiliate

def getLLamaCorpusEmbds(path:str ="/expanse/lustre/projects/csb176/yzound/datasets/msmarco/repllama/dense.dat")->np.memmap:
    embds = np.memmap(path,dtype='float32', mode='r', shape=(MSMARCO_DOC_SZ, 4096))
    return embds

def getTevaronSamples(num_neg:int=1)->list:
    triples = []
    ds = load_dataset("Tevatron/msmarco-passage")["train"]
    for sample in ds:
        if len(sample["negative_passages"]) < num_neg: 
                continue
        triples.append([int(sample["query_id"]),int(sample["positive_passages"][0]["docid"])])
        for i in range(num_neg):
            triples[-1].append(int(sample["negative_passages"][i]["docid"]))
    return triples

def getMSMARCOCorpus(collection:str = "/expanse/lustre/projects/csb176/yzound/datasets/msmarco/collection.tsv")->Dict[int,str]:
    corpus = DefaultDict()
    with open(collection, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            pid = int(pid)
            corpus[pid] = passage
    return corpus

def getMSMARCOQuery(path:str = "/expanse/lustre/projects/csb176/yzound/datasets/msmarco/queries.train.tsv")->Dict[int,str]:
    queries = DefaultDict()
    with open(path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query
    return queries

def getMSMARCOCEscore(path:str = "/expanse/lustre/projects/csb176/yzound/datasets/msmarco/ce_score/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz") -> Dict[int,Dict[int,float]]:
    with gzip.open(path, 'rb') as fIn:
        ce_scores = pickle.load(fIn)
    return ce_scores

def getSpladeHN(queries_texts:Dict[int,str],path:str = "/expanse/lustre/projects/csb185/yifanq/msmarco/train_queries_distill_splade_colbert_0.json"):
    train_queries = dict()
    with open(path) as f: #training_queries_splade_max_156000_ceclean.json
        for line in f:
            qid = line.split("\t")[0]
            train_queries[qid] = json.loads(line.split("\t")[1])
            if len(train_queries[qid]['neg']) == 0 or len(train_queries[qid]['pos']) == 0:
                del train_queries[qid]
                continue
            train_queries[qid]['query'] = queries_texts[int(qid)]
            '''
            remove the negative that has a higher score than positive

            pos_min_ce_score = min([ce_scores[int(qid)][int(pid[1])] for pid in train_queries[qid]['pos']])
            ce_score_threshold = pos_min_ce_score - ce_score_margin
            if args.denoise:
                train_queries[qid]['neg'] = [x for x in train_queries[qid]['neg'] if x[0] <= args.num_negs_per_system and ce_scores[int(qid)][int(x[1])] < ce_score_threshold]
            else:
            '''

            '''
            {qid:, query:"query text", pos:[[?,corpus_idx,?]], neg:[[?,corpus_idx,?]]}
            '''
            train_queries[qid]['pos'] = [x[1] for x in train_queries[qid]['pos']]
            train_queries[qid]['neg'] = [x[1] for x in train_queries[qid]['neg'] if x[1] not in train_queries[qid]['pos']] # pos id might be in the negatives
            
    return train_queries

class MSMARCODataset(Dataset):
    def __init__(self, dataset, queries, corpus, ce_scores, num_of_neg):
        self.queries = queries
        self.corpus = corpus
        self.ce_scores = ce_scores
        self.num_neg = num_of_neg
        self.dataset = dataset

    def __getitem__(self, index):
        sample = self.dataset[index]
        query = self.queries[int(sample['query'])]
        passage = [self.corpus[int(sample['positive'])]]
        scores = [self.ce_scores[int(sample['query'])][int(sample['positive'])]]
        for i in range(self.num_neg):
            passage.append(self.corpus[int(sample[f'negative_{i+1}'])])
            scores.append(self.ce_scores[int(sample['query'])][int(sample[f'negative_{i+1}'])])
        return {'query':query,'passage':passage,'label':0,'ce_scores':np.array(scores,dtype='float32')} #making sure is wont create double

    def __len__(self):
        return len(self.dataset)

'''
Each example in this dataset passes an extra embeddings to the trained model
'''
class MSMARCODatasetExtraEmbedding(Dataset):
    '''
    make sure this is cleaned up
    '''
    def __init__(self, training_samples, corpus, embeddings, ce_scores, num_neg=1, loss_type="marginmse", topk=20, model_type="splade"):
        self.training_samples = training_samples
        self.queries_ids = list(training_samples.keys())
        self.corpus = corpus
        self.embeddings = embeddings
        self.ce_scores = ce_scores
        self.num_neg = num_neg
        self.loss_type = loss_type
        if self.loss_type == "marginmse":
            assert (self.num_neg == 1)

        self.model_type = model_type
                
    def __getitem__(self, item):
        #self.iter_num += 1
        sample = self.training_samples[self.queries_ids[item]]
        if self.model_type == "colbert":
            query_text = "[unused0] " + sample['query']
        else:
            query_text = sample['query']

        qid = sample['qid']
        
        pos_id = sample['pos'].pop(0)  # Pop positive and add at end
        if self.model_type == "colbert":
            pos_text = "[unused1] " + self.corpus[pos_id]
        else:
            pos_text = self.corpus[pos_id]
        sample['pos'].append(pos_id)
        

        pos_score = self.ce_scores[qid][pos_id]
        # Get a negative passage
        neg_texts = []
        neg_scores = []
        if len(sample['neg']) < self.num_neg :
            neg_corpus_idx = np.random.choice(sample['neg'],size=self.num_neg,replace=True)
        else:
            neg_corpus_idx = np.random.choice(sample['neg'],size=self.num_neg,replace=False) #randomly pick some negative samples
        for neg_id in neg_corpus_idx:
            neg_scores.append(self.ce_scores[qid][neg_id])
            if self.model_type == "colbert":
                neg_text = "[unused1] " + self.corpus[neg_id]
            else:
                neg_text = self.corpus[neg_id]
            neg_texts.append(neg_text)
        
        if self.loss_type == "marginmse" or self.loss_type == "marginmseWithDecoder" or self.loss_type == "marginmseFlopSIG":
            return InputExample(texts=[query_text, pos_text, neg_texts[0]], label=pos_score - neg_scores[0], affiliate={'pos':self.embeddings[pos_id],'neg':self.embeddings[neg_corpus_idx[0]]})
        elif self.loss_type == "marginmse_ib":
            return InputExample(texts=[query_text, pos_text, neg_texts[0]], label=[pos_score, neg_scores[0]])
        elif self.loss_type in ["kldiv", "kldiv_focal", "kldiv_ib"]:
            target_score = [pos_score] + neg_scores
            neg_embds = [self.embeddings[idx] for idx in neg_corpus_idx]
            return InputExample(texts=[query_text, pos_text] + neg_texts,
                                label=target_score,affiliate={'pos':self.embeddings[pos_id],'neg':neg_embds})  # length of label is number of texts
        elif self.loss_type == "marginkldiv":
            target_score = torch.tensor([pos_score - neg_score for neg_score in neg_scores])
            target_score = torch.nn.functional.log_softmax(target_score)
            return InputExample(texts=[query_text, pos_text] + neg_texts,
                                label=target_score.tolist())  # length of label is number
            '''
            elif self.loss_type == "marginmse_position":
                return InputExample(texts=[query_text, pos_text, neg_texts[0]],
                                    label=[pos_score - neg_scores[0], pos_idx, neg_idx[0]])
            elif self.loss_type in ["wce", "ce"]:
                return InputExample(texts=[query_text, pos_text] + neg_texts, label=[pos_idx] + neg_idx)
            elif self.loss_type == "marginkldiv_position":
                ce_diffs = [neg_score - pos_score for neg_score in neg_scores]
                target_score = torch.tensor(ce_diffs)
                target_score = torch.nn.functional.log_softmax(target_score)

                ##########weight defined 1 #############
                # alpha = 0.2
                # weights = [alpha/(1+np.exp(neg_i - pos_idx)) + 1 for neg_i in neg_idx]
                # eights = [w if ce_diff > ce_threshold else 1.0 for w,ce_diff in zip(weights,ce_diffs)]
                ##########weight define 2 ##############
                # weights = [np.log10(max(pos_idx - pos_ce_idx, 1))/2 + 1] * len(neg_idx)

                return InputExample(texts=[query_text, pos_text] + neg_texts,
                                    label=target_score.tolist() + [pos_idx] + neg_idx)  # length of label is number
            elif self.loss_type in ["kldiv_position", 'kldiv_position_focal']:
                ce_diffs = [neg_score - pos_score for neg_score in neg_scores]
                target_score = torch.tensor([pos_score] + neg_scores)
                target_score = torch.nn.functional.log_softmax(target_score)
                return InputExample(texts=[query_text, pos_text] + neg_texts,
                                    label=target_score.tolist() + [pos_idx] + neg_idx)  # length of label is number
            elif self.loss_type == "kldiv_position_reverse":
                ce_diffs = [neg_score - pos_score for neg_score in neg_scores]
                target_score = -torch.tensor([pos_score] + neg_scores)
                target_score = torch.nn.functional.log_softmax(target_score)
                return InputExample(texts=[query_text, pos_text] + neg_texts,
                                    label=target_score.tolist() + [pos_idx] + neg_idx)  # length of label is number
            '''
        else:
            raise ("Unrecogized loss type!")
            return

    def __len__(self):
        return len(self.queries_ids)


class MSMARCODatasetTriple(Dataset):
    def __init__(self, queries:dict, corpus,triples):
        self.triples = triples
        self.corpus = corpus
        self.query = queries
    def __getitem__(self, idx):
        qid, pid, nid = self.triples[idx]
        return {'query':self.query[qid],'pos':self.corpus[pid],'neg':self.corpus[nid]} #positive always is the first

    def __len__(self):
        return len(self.triples)
    
