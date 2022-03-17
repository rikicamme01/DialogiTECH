import pandas as pd
df = pd.read_csv('./RepML/data/Union/Hyperion.csv')

import string
import re

def clean_text(text:str) -> str:
    #delete double punctuation
    text =  re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)
    # add space between a word and punctuation
    text = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)
    return text

dataset = []

for row in df.itertuples():
    text = row.Testo
    
    if pd.isna(text):
        sample['Stralci'].append(clean_text(row.Stralcio))
        sample['Repertori'].append(row.Repertorio)

    else:
        sample = {}
        sample['Testo'] = clean_text(text)
        sample['Stralci'] = [clean_text(row.Stralcio)]

        sample['Repertori'] = [row.Repertorio]
        dataset.append(sample)



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
    for span in spans:
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

def find_segmentation_by_bounds(bounds: list, text: str) -> str:
    segmentation = ['0' for i in range(len(text))]
    for bound in bounds:
        if bound[1] < len(text):
            segmentation[bound[1]] = '1'
        else:
            segmentation[-1] = '1'
    return ''.join(segmentation)
    
    

for sample in dataset:
    sample['Bounds'] = find_word_bounds(sample['Stralci'], sample['Testo'])
    sample['Segmentation'] = find_segmentation_by_bounds(sample['Bounds'], sample['Testo'])


import nltk
nltk.download('punkt')
from nltk import sent_tokenize
import re

nltk_pred = []
spans_pred = []

for sample in dataset:
    spans = sent_tokenize(sample['Testo'])
    bounds = []    
    #bounds += find_char_bounds([sample['Testo']], sample['Testo'])
    bounds += find_word_bounds(spans, sample['Testo'])
    nltk_pred.append(bounds)
    spans_pred.append(spans) 


import numpy as np
# A è B sono tupe con i bound dello span
def IoU(A, B):
    if A == B:
        return 1
    start = max(A[0], B[0])
    end = min(A[1], B[1])
    if(start > end):
        return 0
    intersection = end - start
    return intersection / (A[1] - A[0] + B[1] - B[0] - intersection)

def compute_IoUs(pred_bounds, gt_spans):
    """
    Given a list of predicted spans and a list of ground truth spans, 
    compute the IoU between each pair of spans
    
    :param pred_bounds: a tuple of (start, end) denoting the predicted answer
    :param gt_spans: a list of tuples of the form (start, end) representing the spans of each ground
    truth annotation
    :return: a list of IoUs for each ground truth span.
    """
    IoUs = []
    for gt_bounds in gt_spans:
        IoUs.append(IoU(pred_bounds, gt_bounds)) 
    return IoUs

#Input: text_spans_dict = [ {
#           'Bounds' : (a,b), 
#           'IoU' : float,
#           'Repertorio': 'string':
#           } ]
def normalize(text_spans_dict, gt_spans):
    normalized = []
    for i in range(len(text_spans_dict)):
        #normalized is not empty
        if normalized:
            if normalized[-1]['Repertorio'] == text_spans_dict[i]['Repertorio']:
                new_span = (normalized[-1]['Bounds'][0], text_spans_dict[i]['Bounds'][1])
                new_span_features = {
                    'Bounds' : new_span, 
                    'IoU' : None,
                    'Repertorio' : text_spans_dict[i]['Repertorio']
                    }
                del normalized[-1]
                normalized.append(new_span_features)
            else:
                normalized.append(text_spans_dict[i])
        else:
            normalized.append(text_spans_dict[i])
        
    
    for i in range(len(normalized)):
        normalized[i]['IoU'] = max(compute_IoUs(normalized[i]['Bounds'], gt_spans['Bounds']))
    return normalized

def intersection(A, B):
    if A == B:
        return 1
    start = max(A[0], B[0])
    end = min(A[1], B[1])
    if(start > end):
        return 0
    return end - start

def normalize_bounds_by_repertoire(bounds, sample):
    bounds_w_rep = []
    for bound in bounds:
        intersections = []
        for gt_bound in sample['Bounds']:
            intersections.append(intersection(bound, gt_bound))
        rep_idx = np.argmax(intersections)
        bounds_w_rep.append({
            'Bounds': bound,
            'Repertorio': sample['Repertori'][rep_idx]
            })
    normalized = []
    for i in range(len(bounds_w_rep)):
        #normalized is not empty
        if normalized:
            if normalized[-1]['Repertorio'] == bounds_w_rep[i]['Repertorio']:
                new_span = (normalized[-1]['Bounds'][0], bounds_w_rep[i]['Bounds'][1])
                new_span_features = {
                    'Bounds' : new_span, 
                    'Repertorio' : bounds_w_rep[i]['Repertorio']
                    }
                del normalized[-1]
                normalized.append(new_span_features)
            else:
                normalized.append(bounds_w_rep[i])
        else:
            normalized.append(bounds_w_rep[i])
    return [e['Bounds'] for e in normalized]


from nltk.metrics.segmentation import windowdiff, ghd, pk

met_list = []


for i,sample in enumerate(dataset):
    seg_pred = find_segmentation_by_bounds(nltk_pred[i], sample['Testo'])
    
    wd_value = windowdiff(sample['Segmentation'], seg_pred,  6)
    
    ghd_value = ghd(sample['Segmentation'], seg_pred)
    
    pk_value = pk(sample['Segmentation'], seg_pred, 6)

    text_IoUs = []
    for bound in nltk_pred[i]:
        IoUs = compute_IoUs(bound, sample['Bounds'])
        best = np.argmax(IoUs)
        text_IoUs.append(IoUs[best])
    
    met_dict = {
        'windowdiff' : wd_value,
        'ghd' : ghd_value,
        'pk' : pk_value,
        'iou' : text_IoUs
        }
    met_list.append(met_dict)

norm_met_list = []
norm_span_counter = 0

for i,sample in enumerate(dataset):
    norm_pred_bounds = normalize_bounds_by_repertoire(nltk_pred[i], sample)
    norm_span_counter += len(norm_pred_bounds)

    seg_pred = find_segmentation_by_bounds(norm_pred_bounds, sample['Testo'])
    
    wd_value = windowdiff(sample['Segmentation'], seg_pred,  6)
    
    ghd_value = ghd(sample['Segmentation'], seg_pred)
    
    pk_value = pk(sample['Segmentation'], seg_pred, 6)

    text_IoUs = []
    for bound in norm_pred_bounds:
        IoUs = compute_IoUs(bound, sample['Bounds'])
        best = np.argmax(IoUs)
        text_IoUs.append(IoUs[best])
    
    norm_met_dict = {
        'windowdiff' : wd_value,
        'ghd' : ghd_value,
        'pk' : pk_value,
        'iou' : text_IoUs
        }
    norm_met_list.append(norm_met_dict)

print('----------------------------------------------------------')
print('Risultati labels GT e stralci non uniti')

print('Numero testi nel dataset:', str(len(dataset)))

n_spans = 0
for e in dataset:
    n_spans += len(e['Bounds'])
print('Numero stralci nel dataset:', str(n_spans))

n_spans = 0
for e in nltk_pred:
    n_spans += len(e)
print('Numero stralci predetti:', str(n_spans))

IoUs = [e['iou'] for e in met_list]
flat_IoUs = [item for sublist in IoUs for item in sublist]
mean_IoU = np.mean(flat_IoUs)
mean_wd = np.mean([e['windowdiff'] for e in met_list])
mean_pk = np.mean([e['pk'] for e in met_list])
mean_ghd = np.mean([e['ghd'] for e in met_list])

perfect_spans = flat_IoUs.count(1)
print('Percentuale span perfetti: ', str(perfect_spans / len(flat_IoUs)))

print('Media IoU:', str(mean_IoU))
print('Media Windowdiff:', str(mean_wd))
print('Media Pk:', str(mean_pk))
print('Media ghd:', str(mean_ghd))


print('----------------------------------------------------------')
print('Risultati labels GT e stralci uniti')

print('Numero testi nel dataset:', str(len(dataset)))

n_spans = 0
for e in dataset:
    n_spans += len(e['Bounds'])
print('Numero stralci nel dataset:', str(n_spans))

print('Numero stralci predetti:', str(norm_span_counter))

IoUs = [e['iou'] for e in norm_met_list]
flat_IoUs = [item for sublist in IoUs for item in sublist]
mean_IoU = np.mean(flat_IoUs)
mean_wd = np.mean([e['windowdiff'] for e in norm_met_list])
mean_pk = np.mean([e['pk'] for e in norm_met_list])
mean_ghd = np.mean([e['ghd'] for e in norm_met_list])

perfect_spans = flat_IoUs.count(1)

print('Percentuale span perfetti: ', str(perfect_spans / len(flat_IoUs)))

print('Media IoU:', str(mean_IoU))
print('Media Windowdiff:', str(mean_wd))
print('Media Pk:', str(mean_pk))
print('Media ghd:', str(mean_ghd))


from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('MiBo/RepML')
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-uncased")


predicted_dataset = []


for i, span_group in enumerate(spans_pred):
  text_features = {}
  text_features['Testo'] = dataset[i]['Testo']
  text_features['Stralci'] = [span.lower() for span in span_group]
  text_features['Bounds'] = nltk_pred[i]
  predicted_dataset.append(text_features)


import re

import pandas as pd
import torch
from sklearn import preprocessing
from transformers import AutoTokenizer

LABELS = [
                'anticipazione',
                'causa',
                'commento',
                'conferma',
                'considerazione',
                'contrapposizione',
                'deresponsabilizzazione',
                'descrizione',
                'dichiarazione di intenti',
                'generalizzazione',
                'giudizio',
                'giustificazione',
                'implicazione',
                'non risposta',
                'opinione',
                'possibilità',
                'prescrizione',
                'previsione',
                'proposta',
                'ridimensionamento',
                'sancire',
                'specificazione',
                'valutazione'
        ]


def decode_labels(encoded_labels):
    le = preprocessing.LabelEncoder()
    le.fit(LABELS)
    return le.inverse_transform(encoded_labels)


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import utils
import torch
from torch.utils.data import DataLoader

def predict_labels(text: dict)-> list:
  pred = []
  if text['Stralci']:
    encodings = tokenizer(
          text['Stralci'],
          max_length=512,
          add_special_tokens=True,
          return_attention_mask=True,
          padding=True,
          truncation=True,
          return_tensors="pt"
      )
      
  test_dataset = TensorDataset(encodings['input_ids'],encodings['attention_mask'])
  test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
 
  for i, batch in enumerate(test_dataloader):
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      with torch.no_grad():        

          # Forward pass, calculate logits
          # argmax(logits) = argmax(Softmax(logits))
          outputs = model(b_input_ids, 
                                  attention_mask=b_input_mask)
          logits = outputs[0]

      logits = logits.detach().cpu()

      batch_pred = logits.softmax(dim=1)
      pred += batch_pred.argmax(dim=1)
  return pred


# Setup for testing with gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

for i, text in enumerate(predicted_dataset):
  pred = predict_labels(text)
  rep = decode_labels(list(pred))
  predicted_dataset[i]['Repertori'] =rep

  if i%100==0:
    print('testo: ', str(i))

norm_met_list = []
norm_span_counter = 0

for i,sample in enumerate(predicted_dataset):
    norm_pred_bounds = normalize_bounds_by_repertoire(nltk_pred[i], sample)
    norm_span_counter += len(norm_pred_bounds)

    seg_pred = find_segmentation_by_bounds(norm_pred_bounds, sample['Testo'])
    
    wd_value = windowdiff(dataset[i]['Segmentation'], seg_pred,  6)
    
    ghd_value = ghd(dataset[i]['Segmentation'], seg_pred)
    
    pk_value = pk(dataset[i]['Segmentation'], seg_pred, 6)

    text_IoUs = []
    for bound in norm_pred_bounds:
        IoUs = compute_IoUs(bound, dataset[i]['Bounds'])
        best = np.argmax(IoUs)
        text_IoUs.append(IoUs[best])
    
    norm_met_dict = {
        'windowdiff' : wd_value,
        'ghd' : ghd_value,
        'pk' : pk_value,
        'iou' : text_IoUs
        }
    norm_met_list.append(norm_met_dict)

print('----------------------------------------------------------')
print('Risultati labels GT e stralci uniti')

print('Numero testi nel dataset:', str(len(dataset)))

n_spans = 0
for e in dataset:
    n_spans += len(e['Bounds'])
print('Numero stralci nel dataset:', str(n_spans))

print('Numero stralci predetti:', str(norm_span_counter))

IoUs = [e['iou'] for e in norm_met_list]
flat_IoUs = [item for sublist in IoUs for item in sublist]
mean_IoU = np.mean(flat_IoUs)
mean_wd = np.mean([e['windowdiff'] for e in norm_met_list])
mean_pk = np.mean([e['pk'] for e in norm_met_list])
mean_ghd = np.mean([e['ghd'] for e in norm_met_list])

perfect_spans = flat_IoUs.count(1)

print('Percentuale span perfetti: ', str(perfect_spans / len(flat_IoUs)))

print('Media IoU:', str(mean_IoU))
print('Media Windowdiff:', str(mean_wd))
print('Media Pk:', str(mean_pk))
print('Media ghd:', str(mean_ghd))