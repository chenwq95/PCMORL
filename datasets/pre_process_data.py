import csv
import nltk
import pickle

#file_path = "./data/mscoco/paraphrases.dev.tsv"

all_tokens = {}
#base_file_path = "./data/mscoco/"
base_file_path = "./data/quora/"
print("base_file_path:", base_file_path)

def update_token_dict(sen, all_tokens):
    tokens = nltk.word_tokenize(sen.lower())
    for token in tokens:
        if (token in all_tokens.keys()):
            all_tokens[token] += 1
        else:
            all_tokens[token] = 1

for split in ["train", "dev", "test"]:
    file_path = base_file_path + "paraphrases." + split + ".tsv"
    with open(file_path, "r") as data_file:
        for line_num, row in enumerate(csv.reader(data_file, delimiter='\t')):

            source_sequence = row[0]
            target_sequences = list(row[1:])
            #
            update_token_dict(source_sequence, all_tokens)
            for ts in target_sequences:
                update_token_dict(ts, all_tokens)

            #break
            
sub_tokens = []
for k,v in all_tokens.items():
    if (v >= 5):
        sub_tokens.append({"word":k, "count":v})
        #sub_tokens[k] = v
sub_tokens = sorted(sub_tokens, key=lambda k: k["count"], reverse=True)
with open(base_file_path + "coco_vocabulary.txt", "w") as f:
    for token in sub_tokens:
        f.write(token['word'] + "\n")
    f.write('UNK')
    
new_vocab = {item['word']:k+1 for k,item in enumerate(sub_tokens)}

id2word = {v:k for k,v in new_vocab.items()}

print("vocab size: ", len(new_vocab))

import numpy as np
all_lens = []
for split in ["train", "dev", "test"]:
    file_path = base_file_path + "paraphrases." + split + ".tsv"
    with open(file_path, "r") as data_file:
        for line_num, row in enumerate(csv.reader(data_file, delimiter='\t')):
            source_sequence = row[0]
            target_sequences = list(row[1:])
            
            sen = source_sequence
            tokens = nltk.word_tokenize(sen.lower())
            all_lens.append(len(tokens))
            for sen in target_sequences:
                tokens = nltk.word_tokenize(sen.lower())
                all_lens.append(len(tokens))
                

all_lens.sort()
ind = round(len(all_lens) * 0.95)
maxlen = all_lens[ind]
print("maxlen: ", maxlen)

def convert_sent_to_Ids(sen, maxlen):
    tokens = nltk.word_tokenize(sen.lower())
    templen = len(tokens)
    if (templen > maxlen):
        templen = maxlen
    ids = np.zeros(templen, dtype=np.long)
    for i, token in enumerate(tokens[:maxlen]):
        ids[i] = new_vocab.get(token, len(new_vocab)+1)
    return ids


sentIds = {}
for split in ["train", "dev", "test"]:
    file_path = base_file_path + "paraphrases." + split + ".tsv"
    with open(file_path, "r") as data_file:
        for line_num, row in enumerate(csv.reader(data_file, delimiter='\t')):
            source_sequence = row[0]
            target_sequences = list(row[1:])
            target_num = len(target_sequences)
            
            sample_id = split + "_" + str(line_num)
            sentIds[sample_id] = {}
            sentIds[sample_id]["source_seq"] = np.zeros(maxlen, dtype=np.long)
            sentIds[sample_id]["input_seq"] = np.zeros((target_num, maxlen+1), dtype=np.long)
            sentIds[sample_id]["target_seq"] = -1 * np.ones((target_num, maxlen+1), dtype=np.long)

            ids = convert_sent_to_Ids(source_sequence, maxlen)
            sentIds[sample_id]["source_seq"][:len(ids)] = ids
            
            for k, sen in enumerate(target_sequences):
                ids = convert_sent_to_Ids(sen, maxlen)
                templen = len(ids)
                sentIds[sample_id]["input_seq"][k,1:templen+1] = ids
                sentIds[sample_id]["target_seq"][k,:templen] = ids
                sentIds[sample_id]["target_seq"][k,templen] = 0

print("saving sentIds", len(sentIds))
with open(base_file_path + "sentIds.pkl", "wb") as f:
    pickle.dump(sentIds, f)