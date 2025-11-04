import json
import gzip
import numpy as np
import struct
from tqdm import tqdm
import argparse
def convertBinary(num):
    n = int(num)
    return struct.pack('<I', n)

def binarySequence(arr, fout):
    size = len(arr)
    fout.write(convertBinary(size))
    for i in arr:
        fout.write(convertBinary(i))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection',type=str)
    parser.add_argument('--numbers',type=int,default=1,help="number of collections file")
    parser.add_argument('--threshold',type=int,help="indexing threshold",default=0)
    parser.add_argument('--output',type=str,help="output prefix")
    args = parser.parse_args()
    json_path = args.collection

    posting = {}

    length = []
    idx = 0
    for i in range(args.numbers):
        print(i)
        for line in gzip.open("%s%d.jsonl.gz" % (json_path, i)):
            doc_dict = json.loads(line)
            #id = doc_dict['id']

            vector = doc_dict['vector']

            length_t = 0
            for k in vector:
                score = int(vector[k])
                #print(f"{score} {vector[k]}")
                if score > args.threshold:
                    length_t += 1

                    if k not in posting:
                        posting[k] = []

                    posting[k] += [idx, score]
            idx += 1
            
            length.append(length_t)

    term_id = {}
    id = 0
    for k in posting:
        term_id[k] = id
        id += 1

    with open(args.output + '.id', 'w') as f:
        json.dump(term_id, f)

    fout_docs = open(args.output + ".docs", 'wb')
    fout_freqs = open(args.output + ".freqs", 'wb')
    binarySequence([len(length)], fout_docs)


    for k in tqdm(posting):
        binarySequence(posting[k][::2], fout_docs) # docIDs
        binarySequence(posting[k][1::2], fout_freqs) # score instead of freq
    fout_docs.close()
    fout_freqs.close()
    fout_sizes = open(args.output + ".sizes", 'wb')
    binarySequence(length, fout_sizes)
    fout_sizes.close()

    s = 0
    for pos_len in length:
        s += pos_len
    print(f"posting list avg length: {s/len(length)}")