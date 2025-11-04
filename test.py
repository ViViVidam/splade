import json
from collections import defaultdict

def read_qrels(fIn) -> dict:
    qrels = defaultdict(str)
    for line in fIn:
        qid,_,doc,rel = line.strip().split('\t')
        assert(int(rel)==1)
        qrels[qid] = doc
    return qrels

def read_queries(fIn) -> dict:
    queries = defaultdict(str)
    for line in fIn:
        qid,text = line.strip().split('\t')
        # assert(int(rel)==1)
        queries[qid] = text
    return queries

if __name__ == "__main__":
    with open("/projects/bcgk/yzound/datasets/msmarco/qrels.train.tsv","r") as reader:
        qrel = read_qrels(reader)
    with open("/projects/bcgk/yzound/datasets/msmarco/queries.train.tsv","r") as reader:
        queries = read_queries(reader)
    
    in_cnt = 0
    for k in queries.keys():
        if k not in qrel.keys():
            continue
        in_cnt+=1

print(f"{len(list(queries.keys()))} / {in_cnt}")
print(f"{len(list(qrel.keys()))}")