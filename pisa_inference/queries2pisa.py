import json
import sys
import math

term_id = {}
with open(sys.argv[1] + '.id') as f:
    term_id = json.load(f)

# f_in = open('/share/paper-submission/unicoil-index/filtered-msmarco-dev-queries.tsv')
# f_in = open('/home/yingrui/MSMARCO/queries/queries_top1000.dev.tsv.spladev2.weighted')
# f_in = open('/home/yifan/Downloads/pisa/data/queries.unicoil.sorted.tsv')
f_in = open(sys.argv[2])
# f_in = open('deepimpact_short.dev.tsv')
# f_in = open('/nvme/yifanq/msmarco-doc-ggt/msmarco-docdev-queries-tok.tsv')
# f_in = open('/home/yingrui/MSMARCO/queries/msmarco-test2019-queries-qrel.tsv.spladev2.weighted')
# f_in = open('/home/yifan/Downloads/dl1920/msmarco-test2019-queries-qrel-piece.tsv')
# f_in = open('/nvme/yifanq/msmarco-doc-ggt/filtered-msmarco-docdev-queries.tsv')
# f_in = open('/nvme/yifanq/bm25-t5-index/queries.bert.dev.tsv')
# f_in = open('/nvme/yifanq/bm25-t5-index/queries.bert.dev.tsv.all')

# times = {}

thres = 0
query_length = 0
cnt = 0
f_out = open(sys.argv[3], 'w')
for line in f_in:
    cnt += 1
    line = line.strip().split('\t')
    wlist = line[1].split(" ")
    qid = line[0]
    wlist_n = []

    times = {}
    for w in wlist:
        if w in term_id:
            if w not in times:
                times[w] = 0
            times[w] += 1

    #tt = list(times.values())
    #tt.sort()
    #thres = tt[math.floor(len(tt) * (1 - float(sys.argv[4])))]
    thres = int(sys.argv[4])
    
    for w in wlist:
        if w in term_id:
            if times[w] > thres:
                wlist_n.append(str(term_id[w]))
    
    for w in times.keys():
        if times[w] > thres:
            query_length+=1
        # else:
        #     print(wlist[i], "not found in the given docs, assign a new term_id")
        #     term_id[wlist[i]] = id
        #     id += 1
        #     wlist[i] = str(id)
            
    to_write = qid + ': ' + ' '.join(wlist_n)
    f_out.write(to_write)
    f_out.write("\n")
f_in.close()
f_out.close()
print(f"query avg length {query_length/cnt}")