from time import sleep
import copy
from collections import Counter, defaultdict
import re,glob,csv,json
import sys,os
import pickle
import argparse
import os
import json
import numpy as np
from datasets.utils import *
from tqdm import tqdm
from datasets import load_3Deval_dataset
import random
import numpy as np
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import re
sys.path.append(os.path.join(os.getcwd()))


articles = ["a", "an", "the"]

periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(,)(\d)")
punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]
contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (
            re.search(commaStrip, inText) != None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText

def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText


def simple_ratio(numerator,denominator): 
    num_numerator=sum([1 if token in numerator else 0 for token in denominator])
    num_denominator=len(denominator)
    return num_numerator/num_denominator

def tokens_score(ref: str,pred: str)->float:
    return 1. if ref==pred else 0.


def evals_json(gold_data, preds):
    score_list = ['Top1 (EM)']
    score = {s:[] for s in score_list}
    
    for id, item in gold_data.items():
        ref_answers=item
      
        pred_text=preds[id]

        # top-1
        if pred_text[0] in ref_answers:
            score['Top1 (EM)'].append(1)
        else:
            score['Top1 (EM)'].append(0)

    rlt={}
    for k,v in score.items():
        assert len(v)==len(gold_data), len(v)
        rlt[k]=np.mean(v)*100
    return rlt

def eval_pycoco(gold_data, preds, use_spice=False):
    score_list = ['Top1 (EM)','Top1 (F-value)','BLEU-1','BLEU-2','BLEU-3','BLEU-4']
    score = {s:[] for s in score_list}
    
    scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
    ]
    if use_spice:
        scorers.append((Spice(), "SPICE"))
    # =================================================
    # Compute scores
    # =================================================
    rlt={}
    for scorer, method in scorers:
        #eprint('computing %s score...'%(scorer.method()))
       
        score, scores = scorer.compute_score(gold_data, preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
         #       print("%s: %0.3f"%(m, sc*100))
                rlt[m]=sc*100
        else:
          #  print("%s: %0.3f"%(method, score*100))
            rlt[method]=score*100
    return rlt

QT=['All']
def qclass1(question):
    lques = question
    if 'Where' in lques:
        return 'Place'
    if 'How many' in lques:
        return 'Number'
    if 'What color' in lques or 'What is the color' in lques:
        return 'Color'
    if 'What shape' in lques:
        #return 'Shape'
        return 'Object nature'
    if 'What type' in lques:
        #return 'Type'
        return 'Object nature'
    if 'What kind' in lques:
        #return 'Kind'
        return 'Object nature'
    if 'What is' in lques:
        return 'Object'
    return 'Other'

def eval_func(dataset, pred_data):
    scores = {}
    golds = {}
    preds = {}
    id = 0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        golds[id] = gt['gt_choices']
        preds[id] = [pred['text'][0].split("\n")[0]]
        id += 1

    score=evals_json(golds,preds)
    score2=eval_pycoco(golds, preds)
    score.update(score2)
    scores = score

    print (scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="ScanQA_v1.0_val_3dmit")
    parser.add_argument('--answer-file', default="./ckpt/answers_ep1/re1129VQA_ScanQA_v1.0_val_3dmit.json")
    parser.add_argument('--base-data-path', default="./src/data/3D_Benchmark")
    args = parser.parse_args()
   
    dataset_name = args.dataset_name 
    dataset = load_3Deval_dataset(
        args.base_data_path,
        dataset_name,
        'common',
        batch_size = 1
    ).dataset

    task_name = dataset.task_name

    if args.answer_file.endswith('.json'):
        pred_data = json.load(open(args.answer_file,'rb'))
    else:
        file_ext = '.json'
        file_name = task_name + '_' + dataset_name + file_ext
        args.answer_file = os.path.join(args.answer_file, file_name)
        pred_data = json.load(open(args.answer_file, 'rb'))

    print(f'Eval [{args.answer_file}] on {dataset_name}')
    eval_func(dataset, pred_data)