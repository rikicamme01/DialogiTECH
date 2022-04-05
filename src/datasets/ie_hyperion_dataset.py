import string
import re

import torch
from transformers import AutoTokenizer
import numpy as np
import pandas as pd


class IEHyperionDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        dataset = []

        for row in df.itertuples():
            sample = {}
            sample['Testo'] = clean_text(row.Testo)
            sample['Stralci'] = [clean_text(s) for s in row.Stralci]
            sample['Repertori'] = row.Repertori
            sample['Char_Bounds'] = find_char_bounds(sample['Stralci'], sample['Testo'])
            sample['Bounds'] = find_word_bounds(sample['Stralci'], sample['Testo'])
            sample['Char_Segmentation'] = find_segmentation_by_bounds(sample['Char_Bounds'])
            sample['Segmentation'] = find_segmentation_by_bounds(sample['Bounds'])

            dataset.append(sample)
                 

        IE_dict = {
            'Testo': [sample['Testo'] for sample in dataset],
            'Char_Segmentation': [sample['Char_Segmentation'] for sample in dataset],
            'Segmentation': [sample['Segmentation'] for sample in dataset],
            'Bounds': [sample['Bounds'] for sample in dataset],
            'Char_Bounds': [sample['Char_Bounds'] for sample in dataset],
            'Repertori': [sample['Repertori'] for sample in dataset],
            'Stralci': [sample['Stralci'] for sample in dataset]
        }
        self.df = pd.DataFrame(IE_dict)


            

    def __getitem__(self, idx):
        text = self.df['Testo'].iloc[idx]
        encoding = self.tokenizer(text,
                                  # is_pretokenized=True,
                                  return_special_tokens_mask=True,
                                  return_offsets_mapping=True,
                                  add_special_tokens=True,
                                  return_attention_mask=True,
                                  padding='max_length',
                                  truncation=True,
                                  )
        char_labels = list(self.df['Char_Segmentation'].iloc[idx])
        ends = [i for i in range(len(char_labels)) if char_labels[i] == '1']

        last_token_idx = max(index for index, item in enumerate(encoding['special_tokens_mask']) if item == 0)

        encoded_labels = np.ones(len(encoding['input_ids']), dtype=int) * -100
        x = ends.pop(0)
        for i,e in enumerate(encoding['offset_mapping']):
            if e[1] != 0:
                # overwrite label
                if x >= e[0] and x <= e[1]:# Doubt if insert < e[1] because of offset mapping composition
                    encoded_labels[i] = 1
                    if ends: 
                        x = ends.pop(0)
                    else:
                        x = -1
                else:
                    encoded_labels[i] = 0

        encoded_labels[last_token_idx] = 1


        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        return item

    def __len__(self):
        return len(self.df.index)


#Find bounds starting froma text
def find_char_bounds(spans: list, text: str) -> list:
    '''
    Given a list of spans and a text, find the start and end indices of each span in the text.
    Indeces are computed counting CHARS
    
    :param spans: a list of strings to search for
    :type spans: list
    :param text: the text to search
    :type text: str
    :return: A list of tuples, where each tuple contains the start and end index of a span.
    '''
    start = 0
    bounds = []
    last_char = -1
    for span in spans:
        start = text.find(span)
        if start == -1:
            start = last_char + 1
        last_char = start + len(span)
        bounds.append((start, last_char))
        
    return bounds


def find_word_bounds(spans: list, text: str) -> list:
    '''
    Given a list of spans and a text, find the start and end indices of each span in the text.
    Indeces are computed counting WORDS.

    :param spans: a list of strings, each string is a span of text
    :type spans: list
    :param text: the text to be searched
    :type text: str
    :return: A list of tuples, where each tuple is the start and end index of a word in the text.
    '''
    bounds = []
    end = 0
    for span in spans:
        s = span.translate(str.maketrans('', '', string.punctuation))
        word_list = s.split()
        if word_list:   
            text_list = text.translate(str.maketrans('', '', string.punctuation)).split()
            try:
                start = text_list.index(word_list[0], end)
            except:
                if not bounds:
                    start = 0
                else:

                    start = bounds[-1][1] + 1
            end = start + len(word_list) - 1

            bounds.append((start, end))
    return bounds

def find_segmentation(bounds, text):
    text_list = text.translate(str.maketrans('', '', string.punctuation)).split()
    segmentation = ['0' for i in range(len(text_list))]
    segmentation[-1] = '1'
    
    ends = []
    end = 0
    for span in text_list:
        word_list = span.translate(str.maketrans('', '', string.punctuation)).split()
        try:
            end = text_list.index(word_list[-1], end)
        except:
                end = end + len(word_list) -1
        if end < len(text_list):
            ends.append(end)
    for i in ends:
        segmentation[i] = '1'
    
    return ''.join(segmentation)

def find_segmentation_by_bounds(bounds: list) -> str:
    segmentation = ['0' for i in range(bounds[-1][1] + 1)]
    for bound in bounds:
        if bound[1] < len(segmentation):
            segmentation[bound[1]] = '1'
    segmentation[-1] = '1'
    return ''.join(segmentation)

def clean_text(text:str) -> str:
    #delete \n
    text = text.replace('\n', ' ')
    #delete double punctuation
    text =  re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)
    # add space between a word and punctuation
    text = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)    
    return text

def train_val_split(df, tok_name):
    train_size = 0.8
    train_df = df.sample(frac=train_size)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    return IEHyperionDataset(train_df, tok_name), IEHyperionDataset(val_df, tok_name)