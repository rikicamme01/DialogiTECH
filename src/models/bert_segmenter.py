
from collections import deque

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

import pandas as pd 


class BertSegmenter():
    def __init__(self):
        self.model = AutoModelForTokenClassification.from_pretrained('MiBo/SegBert')
        self.tokenizer = AutoTokenizer.from_pretrained('MiBo/SegBert')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text:str) -> list[str]:
        encoded_text = self.tokenizer(text,
                                    return_special_tokens_mask=True,
                                    return_offsets_mapping=True,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors = 'pt'
                                    )
        
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)
        logits = self.model(input_ids, attention_mask)['logits']
        
        full_probs = logits.softmax(dim=-1)
        full_probs = full_probs.view(full_probs.size(1), full_probs.size(2))

        full_labels = decode_segmentation(full_probs, 0.5)
        active_labels = extract_active_preds(full_labels, encoded_text['special_tokens_mask'][0].tolist())

        x = split_by_prediction(active_labels, encoded_text['input_ids'][0].tolist(), encoded_text['offset_mapping'][0].tolist(), text, self.tokenizer)
        return x

def extract_active_preds(preds:list, special_tokens:list) -> deque:
    active = []
    for i, e in enumerate(special_tokens):
        if(e == 0):
            active.append(preds[i])
    active[-1] = 1
    return active
         

def decode_segmentation(probs, threshold):  #one sample
    if threshold < 0 or threshold > 1:
        return None
    segmentation = []
    for prob in probs:
        if prob[1] >= threshold:
            segmentation.append(1)
        else:
            segmentation.append(0)
    segmentation[-1] = 1
    return segmentation

def split_by_prediction(pred:list, input_ids:list, offset_mapping:list, text:str, tokenizer) -> list:
    subword_flags = [False for i in range(len(input_ids))]
    for i in  range(len(input_ids)):
        if offset_mapping[i][1] != 0:
            if tokenizer.decode(input_ids[i])[:2] == '##':
                subword_flags[i] = True
    
    for i in range(len(pred)-1):
        if pred[i] == 1 and subword_flags[i]:
                pred[i] = 0
                pred[i + 1] = 1

    spans = []
    start = 0
    j=0
    for i in range(len(offset_mapping)):
        if offset_mapping[i][1] != 0:
            x = pred[j]
            j += 1
            if x == 1:
                spans.append(text[start:offset_mapping[i][1]])
                start = offset_mapping[i][1]
    if not spans:
        spans.append(text)
    return spans

def normalize(bounds:list, reps:list):
    norm_bounds = []
    norm_reps = []
    
    for i in range(len(bounds)):
        if norm_reps and norm_reps[-1] == reps[i]:
            norm_bounds[-1] = (norm_bounds[-1][0], bounds[i][1])
        else:
            norm_bounds.append(bounds[i])
            norm_reps.append(reps[i])
    return pd.Series([norm_bounds, norm_reps])

    