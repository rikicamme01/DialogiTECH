
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

class BertSegmenter():
    def __init__(self):
        self.model = AutoModelForTokenClassification.from_pretrained('MiBo/SegBert')
        self.tokenizer = AutoTokenizer.from_pretrained('MiBo/SegBert')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text:str):
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

        labels = decode_segmentation(full_probs, 0.5)
        x = split_by_prediction(labels, encoded_text['input_ids'][0].tolist(), encoded_text['offset_mapping'][0].tolist(), text, self.tokenizer)
        return x

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
"""
def split_by_prediction(pred:list, input:dict, text:str, tokenizer) -> list:
    offset_mapping = input['offset_mapping'][0].tolist()
    i=0
    subword_flags = []
    while i < len(offset_mapping):
        if offset_mapping[i][1] != 0:
            if tokenizer.decode(input['input_ids'][0][i])[:2] == '##':
                subword_flags.append(True)
            else:
                subword_flags.append(False)
        i+=1
        
    for i in range(len(pred)-1):
        if pred[i] == 1:
            if subword_flags[i + 1]:
                pred[i] = 0
                pred[i + 1] =1
        
    spans = []
    start = 0
    i=0
    while i < len(offset_mapping):
        if offset_mapping[i][1] != 0:
            x = pred.pop(0)
            if x == 1:
                spans.append(text[start:offset_mapping[i][1]])
                start = offset_mapping[i][1]
        i+=1
    return spans
"""
def split_by_prediction(pred:list, input_ids:list, offset_mapping:list, text:str, tokenizer) -> list:
    subword_flags = [False for i in range(len(input_ids))]
    tokens = [tokenizer.decode(id) for id in input_ids]
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
    for i in range(len(offset_mapping)):
        if offset_mapping[i][1] != 0:
            x = pred[i]
            if x == 1:
                spans.append(text[start:offset_mapping[i][1]])
                start = offset_mapping[i][1]
    return spans

    