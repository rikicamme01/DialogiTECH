import pandas as pd
import string
import nltk
nltk.download('punkt')
from nltk import sent_tokenize
import re
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn import preprocessing
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import utils
from torch.utils.data import DataLoader




df = pd.read_csv('./RepML/data/Original_csv/Hyperion.csv', na_filter=False)

dataset = []
sample = {}

for row in df.itertuples():
    text = row.Testo
    if text and len(text) > 4:
        dataset.append(sample)
        sample = {}
        sample['Testo'] = text
        sample['Stralci'] = [row.Stralcio]
        sample['Repertori'] = [row.Repertorio]
        
    else:
        sample['Stralci'].append(row.Stralcio)
        sample['Repertori'].append(row.Repertorio)
del dataset[0]

#Find bounds starting froma text
def find_char_bounds(spans: list, text: str) -> list:
    bounds = []
    last_char = 0
    for span in spans:
        start = text.find(span)
        if start == -1:
            start = last_char + 1
        bounds.append((start, start + len(span)))
        last_char = start + len(span)
    return bounds

def find_word_bounds(spans: list, text: str) -> list:
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


for sample in dataset:
    #sample['Bounds'] = find_char_bounds(sample['Stralci'], sample['Testo'])
    sample['Bounds'] = find_word_bounds(sample['Stralci'], sample['Testo'])


nltk_pred = []
spans_pred = []

for sample in dataset:
    tokens = sent_tokenize(sample['Testo'])
    spans = []
    bounds = []
    for x in tokens:
        #spans += re.findall('.*?[.:!?;,]', x)
        spans += re.split('[.:!?;,]', x)
        spans = list(filter(None, spans)) # filter empty strings

    #bounds += find_char_bounds(spans, sample['Testo'])
    bounds += find_word_bounds(spans, sample['Testo'])
    nltk_pred.append(bounds)
    spans_pred.append(spans) 


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
    

metrics = []
normalized_metrics = []
for i, pred_bounds in enumerate(nltk_pred):
    text_IoUs = []
    for pred_span in pred_bounds:
        IoUs = compute_IoUs(pred_span, dataset[i]['Bounds'])
        best = np.argmax(IoUs)
        span_features = {
            'Bounds' : pred_span, 
            'IoU' : IoUs[best],
            'Repertorio' : dataset[i]['Repertori'][best]
            }

        text_IoUs.append(span_features)
    metrics.append(text_IoUs)
    normalized_metrics.append(normalize(text_IoUs, dataset[i]))


print('----------------------------------------------------------')
print('Risultati labels GT e stralci non uniti')


n_spans = 0
for e in dataset:
    n_spans += len(e['Bounds'])
print('Numero stralci nel dataset:', str(n_spans))

n_spans = 0
for e in metrics:
    n_spans += len(e)
print('Numero stralci predetti:', str(n_spans))

mean = 0
long_spans = 0
min_lenght = 0
perfect_spans =0
for text in metrics:
    for span in text:
        if span['Bounds'][1] - span['Bounds'][0] >= min_lenght:
            long_spans += 1
            mean += span['IoU']
            if span['IoU'] == 1:
                perfect_spans += 1
perfect_spans_perc = perfect_spans / long_spans
mean_IoU = mean / long_spans
print('Numero stralci con lunghezza minima = ', str(min_lenght), ': ', str(long_spans))
print('Media IoU:', str(mean_IoU))
print('Percentuale span perfetti: ', str(perfect_spans_perc))



print('----------------------------------------------------------')
print('Risultati labels GT e stralci uniti')


n_spans = 0
for e in dataset:
    n_spans += len(e['Bounds'])
print('Numero stralci nel dataset:', str(n_spans))

n_spans = 0
for e in normalized_metrics:
    n_spans += len(e)
print('Numero stralci predetti:', str(n_spans))

mean = 0
long_spans = 0
min_lenght = 0
perfect_spans =0
for text in normalized_metrics:
    for span in text:
        if span['Bounds'][1] - span['Bounds'][0] >= min_lenght:
            long_spans += 1
            mean += span['IoU']
            if span['IoU'] == 1:
                perfect_spans += 1
perfect_spans_perc = perfect_spans / long_spans
mean_IoU = mean / long_spans
print('Numero stralci con lunghezza minima = ', str(min_lenght), ': ', str(long_spans))
print('Media IoU:', str(mean_IoU))
print('Percentuale span perfetti: ', str(perfect_spans_perc))


#--------------SPAN CLASSIFICATION-------------

model = AutoModelForSequenceClassification.from_pretrained('MiBo/RepML')
tokenizer = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")

predicted_dataset = []
for i, span_group in enumerate(spans_pred):
  text_features = {}
  text_features['Testo'] = dataset[i]['Testo']
  text_features['Stralci'] = [span.lower() for span in span_group]
  text_features['Bounds'] = nltk_pred[i]
  predicted_dataset.append(text_features)

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

# Setup for testing with gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()


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

for i, text in enumerate(predicted_dataset):
  pred = predict_labels(text)
  rep = decode_labels(list(pred))
  predicted_dataset[i]['Repertori'] =rep

  if i%100==0:
    print('testo: ', str(i))

metrics = []
normalized_metrics = []
for i, sample in enumerate(predicted_dataset):
    text_IoUs = []
    for j, pred_bounds in enumerate(sample['Bounds']):
        IoUs = compute_IoUs(pred_bounds, dataset[i]['Bounds'])
        best = np.argmax(IoUs)
        span_features = {
            'Bounds' : pred_bounds, 
            'IoU' : IoUs[best],
            'Repertorio' : predicted_dataset[i]['Repertori'][j]
            }

        text_IoUs.append(span_features)
    metrics.append(text_IoUs)
    normalized_metrics.append(normalize(text_IoUs, dataset[i]))


print('----------------------------------------------------------')
print('Risultati labels predette e stralci NON uniti')


n_spans = 0
for e in dataset:
    n_spans += len(e['Bounds'])
print('Numero stralci nel dataset:', str(n_spans))

n_spans = 0
for e in metrics:
    n_spans += len(e)
print('Numero stralci predetti:', str(n_spans))

mean = 0
long_spans = 0
min_lenght = 0
perfect_spans =0
for text in metrics:
    for span in text:
        if span['Bounds'][1] - span['Bounds'][0] >= min_lenght:
            long_spans += 1
            mean += span['IoU']
            if span['IoU'] == 1:
                perfect_spans += 1
perfect_spans_perc = perfect_spans / long_spans
mean_IoU = mean / long_spans
print('Numero stralci con lunghezza minima = ', str(min_lenght), ': ', str(long_spans))
print('Media IoU:', str(mean_IoU))
print('Percentuale span perfetti: ', str(perfect_spans_perc))


print('----------------------------------------------------------')
print('Risultati labels predette e stralci uniti')

n_spans = 0
for e in dataset:
    n_spans += len(e['Bounds'])
print('Numero stralci nel dataset:', str(n_spans))

n_spans = 0
for e in normalized_metrics:
    n_spans += len(e)
print('Numero stralci predetti:', str(n_spans))

mean = 0
long_spans = 0
min_lenght = 0
perfect_spans =0
for text in normalized_metrics:
    for span in text:
        if span['Bounds'][1] - span['Bounds'][0] >= min_lenght:
            long_spans += 1
            mean += span['IoU']
            if span['IoU'] == 1:
                perfect_spans += 1
perfect_spans_perc = perfect_spans / long_spans
mean_IoU = mean / long_spans
print('Numero stralci con lunghezza minima = ', str(min_lenght), ': ', str(long_spans))
print('Media IoU:', str(mean_IoU))
print('Percentuale span perfetti: ', str(perfect_spans_perc))


