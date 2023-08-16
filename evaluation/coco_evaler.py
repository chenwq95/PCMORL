import os
import sys
import tempfile
import json
from json import encoder
from lib.config import cfg

sys.path.append(cfg.INFERENCE.COCO_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

class COCOEvaler(object):
    def __init__(self, annfile, dump_only=False):
        super(COCOEvaler, self).__init__()
        self.dump_only = (dump_only or (annfile==''))
        if (not self.dump_only):
            self.coco = COCO(annfile)
        if not os.path.exists(cfg.TEMP_DIR):
            os.mkdir(cfg.TEMP_DIR)
        
        self.results_path = "gen_captions/gen.json"

    def eval(self, result):
        if (True):
            in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=cfg.TEMP_DIR)
            json.dump(result, in_file)
            in_file.close()
            print("cfg.TEMP_DIR", cfg.TEMP_DIR)
            print("in_file.name", in_file.name)
        
            cocoRes = self.coco.loadRes(in_file.name)
            cocoEval = COCOEvalCap(self.coco, cocoRes)
            cocoEval.evaluate()
            os.remove(in_file.name)
            return cocoEval.eval
        else:
            #print(result)
            with open(self.results_path, "w") as f:
                json.dump(result, f)
            print("results in", self.results_path, "done")
        
            return None