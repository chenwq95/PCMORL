import csv
import nltk
import pickle
import json

#file_path = "./data/mscoco/paraphrases.dev.tsv"

all_tokens = {}
#base_file_path = "./data/mscoco/"
base_file_path = "./data/quora/"
print("base_file_path:", base_file_path)



for split in ["dev", "test"]:
    file_path = base_file_path + "paraphrases." + split + ".tsv"
    new_obj = {}
    new_obj = dict()
    new_obj['info'] = ''
    new_obj['licenses'] = ''
    new_obj['annotations'] = []
    new_obj['images'] = []

    with open(file_path, "r") as data_file:
        for line_num, row in enumerate(csv.reader(data_file, delimiter='\t')):

            sample_id = split + "_" + str(line_num)
            new_obj['images'].append({'id' : sample_id})
            source_sequence = row[0]
            target_sequences = list(row[1:])
            #
            for i,ts in enumerate(target_sequences):
                tempv = {}
                tempv['image_id'] = sample_id
                tempv['id'] = sample_id + '_' + str(i)
                tempv['caption'] = ts.lower()
                new_obj['annotations'].append(tempv)

            #break
            
            
    annfile = base_file_path + "annotations_"+split+".json"
    with open(annfile, "w") as f:
        json.dump(new_obj, f)
    
