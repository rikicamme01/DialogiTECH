# %%
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]) + '/src')
from ast import literal_eval
from datasets.ie_hyperion_dataset import find_word_bounds, clean_text

df = pd.read_csv('../data/processed/pipeline/test/ie_hyperion.csv', converters={'Stralci': literal_eval, 'Repertori': literal_eval})
#df = df.head(50)
df['Testo'] = df['Testo'].map(clean_text)
df['Stralci'] = df['Stralci'].map(lambda x: [clean_text(s) for s in x])
df['Bounds'] = df.apply(lambda x: find_word_bounds(x['Stralci'], x['Testo']), axis=1).values.tolist()

# %%
def intersection(A, B):
    start = max(A[0], B[0])
    end = min(A[1], B[1])
    if(start > end):
        return 0
    return end - start + 1


def C(pred_bound:tuple, gt_bound:tuple, pred_rep:str, gt_rep:str, norm_factor:int) -> float:
    if pred_rep != gt_rep:
        return 0
    x = intersection(pred_bound, gt_bound)
    return x / norm_factor

def precision(pred_bounds:list, gt_bounds:list, pred_reps:list, gt_reps:list) -> float:
    curr_sum = 0
    for i in range(len(pred_bounds)):
        for j in range(len(gt_bounds)):
            curr_sum += C(pred_bounds[i], gt_bounds[j], pred_reps[i], gt_reps[j], pred_bounds[i][1] - pred_bounds[i][0] + 1)
    return curr_sum / len(pred_bounds)

def recall(pred_bounds:list, gt_bounds:list, pred_reps:list, gt_reps:list) -> float:
    curr_sum = 0
    for i in range(len(pred_bounds)):
        for j in range(len(gt_bounds)):
            curr_sum += C(pred_bounds[i], gt_bounds[j], pred_reps[i], gt_reps[j], gt_bounds[j][1] - gt_bounds[j][0] + 1)
    return curr_sum / len(gt_bounds)

def f1(prec:float, rec:float) -> float:
    if prec and rec:
        return 2 * ((prec * rec)/(prec + rec))
    return 0

def IoU(pred_bounds:list, gt_bounds:list, pred_reps:list, gt_reps:list) -> float:
    curr_sum = 0
    for i in range(len(pred_bounds)):
        for j in range(len(gt_bounds)):
            curr_sum += C(pred_bounds[i], gt_bounds[j], pred_reps[i], gt_reps[j], max(pred_bounds[i][1], gt_bounds[j][1]) - min(pred_bounds[i][0], gt_bounds[j][0]) + 1)
    return curr_sum / len(pred_bounds)



# %%
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

# %% [markdown]
# ## NLTK + BERT

# %%
from models.nltk_segmenter import NLTKSegmenter


nltk_seg = NLTKSegmenter()
df['Stralci_predetti'] = df['Testo'].map(nltk_seg.predict).values.tolist()
df['Bounds_predetti'] = df.apply(lambda x: find_word_bounds(x['Stralci_predetti'], x['Testo']), axis=1).values.tolist()

# %%
from models.bert_rep import BertRep

bert_rep = BertRep()
df['Repertori_predetti'] = df['Stralci_predetti'].map(bert_rep.predict).values.tolist()

# %%
df['Precision'] =  df.apply(lambda x: precision(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['Recall'] =  df.apply(lambda x: recall(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['F1'] =  df.apply(lambda x: f1(x['Precision'], x['Recall']), axis=1)
df['IoU'] =  df.apply(lambda x: IoU(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)

print('NLTK + BERT')
print(df['Precision'].mean())
print(df['Recall'].mean())
print(df['F1'].mean())
print(df['IoU'].mean())

# %%
df[['Norm_bounds', 'Norm_rep']] =  df.apply(lambda x: normalize(x['Bounds_predetti'], x['Repertori_predetti']), axis=1)

df['Norm_precision'] =  df.apply(lambda x: precision(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_recall'] =  df.apply(lambda x: recall(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_f1'] =  df.apply(lambda x: f1(x['Norm_precision'], x['Norm_recall']), axis=1)
df['Norm_IoU'] =  df.apply(lambda x: IoU(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)

print('NLTK + BERT norm')
print(df['Norm_precision'].mean())
print(df['Norm_recall'].mean())
print(df['Norm_f1'].mean())
print(df['Norm_IoU'].mean())

# %% [markdown]
# ## BERT + BERT

# %%
from models.bert_segmenter import BertSegmenter

bert_seg = BertSegmenter()
df['Stralci_predetti'] = df['Testo'].map(bert_seg.predict).values.tolist()
df['Bounds_predetti'] = df.apply(lambda x: find_word_bounds(x['Stralci_predetti'], x['Testo']), axis=1).values.tolist()

# %%
from models.bert_rep import BertRep

bert_rep = BertRep()
df['Repertori_predetti'] = df['Stralci_predetti'].map(bert_rep.predict).values.tolist()

# %%
df['Precision'] =  df.apply(lambda x: precision(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['Recall'] =  df.apply(lambda x: recall(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['F1'] =  df.apply(lambda x: f1(x['Precision'], x['Recall']), axis=1)
df['IoU'] =  df.apply(lambda x: IoU(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)

print('BERT + BERT')
print(df['Precision'].mean())
print(df['Recall'].mean())
print(df['F1'].mean())
print(df['IoU'].mean())

# %%
df[['Norm_bounds', 'Norm_rep']] =  df.apply(lambda x: normalize(x['Bounds_predetti'], x['Repertori_predetti']), axis=1)

df['Norm_precision'] =  df.apply(lambda x: precision(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_recall'] =  df.apply(lambda x: recall(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_f1'] =  df.apply(lambda x: f1(x['Norm_precision'], x['Norm_recall']), axis=1)
df['Norm_IoU'] =  df.apply(lambda x: IoU(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)

print('BERT + BERT norm')
print(df['Norm_precision'].mean())
print(df['Norm_recall'].mean())
print(df['Norm_f1'].mean())
print(df['Norm_IoU'].mean())

# %% [markdown]
# ## NLTK + CLS ottimo

# %%
from models.nltk_segmenter import NLTKSegmenter


nltk_seg = NLTKSegmenter()
df['Stralci_predetti'] = df['Testo'].map(nltk_seg.predict).values.tolist()
df['Bounds_predetti'] = df.apply(lambda x: find_word_bounds(x['Stralci_predetti'], x['Testo']), axis=1).values.tolist()

# %%
import numpy as np
def optimal_rep(pred_bounds: list, gt_bounds:list, reps:list):
    opt_reps = []
    for b in pred_bounds:
        opt = np.argmax([intersection(b, x) for x in gt_bounds])
        opt_reps.append(reps[opt])
    return opt_reps


df['Repertori_predetti'] = df.apply(lambda x: optimal_rep(x['Bounds_predetti'], x['Bounds'], x['Repertori']), axis=1).values.tolist()

# %%
df['Precision'] =  df.apply(lambda x: precision(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['Recall'] =  df.apply(lambda x: recall(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['F1'] =  df.apply(lambda x: f1(x['Precision'], x['Recall']), axis=1)
df['IoU'] =  df.apply(lambda x: IoU(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)

print('NLTK + CLS ottimo')
print(df['Precision'].mean())
print(df['Recall'].mean())
print(df['F1'].mean())
print(df['IoU'].mean())

# %%
df[['Norm_bounds', 'Norm_rep']] =  df.apply(lambda x: normalize(x['Bounds_predetti'], x['Repertori_predetti']), axis=1)

df['Norm_precision'] =  df.apply(lambda x: precision(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_recall'] =  df.apply(lambda x: recall(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_f1'] =  df.apply(lambda x: f1(x['Norm_precision'], x['Norm_recall']), axis=1)
df['Norm_IoU'] =  df.apply(lambda x: IoU(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)

print('NLTK + CLS ottimo norm')
print(df['Norm_precision'].mean())
print(df['Norm_recall'].mean())
print(df['Norm_f1'].mean())
print(df['Norm_IoU'].mean())

# %% [markdown]
# ## BERT + CLS ottimo

# %%
from models.bert_segmenter import BertSegmenter

bert_seg = BertSegmenter()
df['Stralci_predetti'] = df['Testo'].map(bert_seg.predict).values.tolist()
df['Bounds_predetti'] = df.apply(lambda x: find_word_bounds(x['Stralci_predetti'], x['Testo']), axis=1).values.tolist()

# %%
import numpy as np
def optimal_rep(pred_bounds: list, gt_bounds:list, reps:list):
    opt_reps = []
    for b in pred_bounds:
        opt = np.argmax([intersection(b, x) for x in gt_bounds])
        opt_reps.append(reps[opt])
    return opt_reps


df['Repertori_predetti'] = df.apply(lambda x: optimal_rep(x['Bounds_predetti'], x['Bounds'], x['Repertori']), axis=1).values.tolist()

# %%
df['Precision'] =  df.apply(lambda x: precision(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['Recall'] =  df.apply(lambda x: recall(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['F1'] =  df.apply(lambda x: f1(x['Precision'], x['Recall']), axis=1)
df['IoU'] =  df.apply(lambda x: IoU(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)

print('BERT + CLS ottimo')

print(df['Precision'].mean())
print(df['Recall'].mean())
print(df['F1'].mean())
print(df['IoU'].mean())

# %%
df[['Norm_bounds', 'Norm_rep']] =  df.apply(lambda x: normalize(x['Bounds_predetti'], x['Repertori_predetti']), axis=1)

df['Norm_precision'] =  df.apply(lambda x: precision(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_recall'] =  df.apply(lambda x: recall(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_f1'] =  df.apply(lambda x: f1(x['Norm_precision'], x['Norm_recall']), axis=1)
df['Norm_IoU'] =  df.apply(lambda x: IoU(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)

print('BERT + CLS ottimo norm')
print(df['Norm_precision'].mean())
print(df['Norm_recall'].mean())
print(df['Norm_f1'].mean())
print(df['Norm_IoU'].mean())

# %% [markdown]
# ## NLTK + CLS random

# %%
from models.nltk_segmenter import NLTKSegmenter


nltk_seg = NLTKSegmenter()
df['Stralci_predetti'] = df['Testo'].map(nltk_seg.predict).values.tolist()
df['Bounds_predetti'] = df.apply(lambda x: find_word_bounds(x['Stralci_predetti'], x['Testo']), axis=1).values.tolist()

# %%
from random import randint
from random import seed
from datasets.hyperion_dataset import decode_labels

seed(1464)

def rand_cls(n:int) -> list:
    cls = [randint(0,22) for _ in range(n)]
    return decode_labels(cls)

df['Repertori_predetti'] = df.apply(lambda x: rand_cls(len(x['Bounds_predetti'])), axis=1).values.tolist()

# %%
df['Precision'] =  df.apply(lambda x: precision(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['Recall'] =  df.apply(lambda x: recall(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['F1'] =  df.apply(lambda x: f1(x['Precision'], x['Recall']), axis=1)
df['IoU'] =  df.apply(lambda x: IoU(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)

print('NLTK + CLS random')
print(df['Precision'].mean())
print(df['Recall'].mean())
print(df['F1'].mean())
print(df['IoU'].mean())

# %%
df[['Norm_bounds', 'Norm_rep']] =  df.apply(lambda x: normalize(x['Bounds_predetti'], x['Repertori_predetti']), axis=1)

df['Norm_precision'] =  df.apply(lambda x: precision(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_recall'] =  df.apply(lambda x: recall(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_f1'] =  df.apply(lambda x: f1(x['Norm_precision'], x['Norm_recall']), axis=1)
df['Norm_IoU'] =  df.apply(lambda x: IoU(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)

print('NLTK + CLS random norm')
print(df['Norm_precision'].mean())
print(df['Norm_recall'].mean())
print(df['Norm_f1'].mean())
print(df['Norm_IoU'].mean())

# %% [markdown]
# ## BERT + CLS random

# %%
from models.bert_segmenter import BertSegmenter

bert_seg = BertSegmenter()
df['Stralci_predetti'] = df['Testo'].map(bert_seg.predict).values.tolist()
df['Bounds_predetti'] = df.apply(lambda x: find_word_bounds(x['Stralci_predetti'], x['Testo']), axis=1).values.tolist()

# %%
from random import randint
from random import seed
from datasets.hyperion_dataset import decode_labels

seed(1464)

def rand_cls(n:int) -> list:
    cls = [randint(0,22) for _ in range(n)]
    return decode_labels(cls)

df['Repertori_predetti'] = df.apply(lambda x: rand_cls(len(x['Bounds_predetti'])), axis=1).values.tolist()

# %%
df['Precision'] =  df.apply(lambda x: precision(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['Recall'] =  df.apply(lambda x: recall(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['F1'] =  df.apply(lambda x: f1(x['Precision'], x['Recall']), axis=1)
df['IoU'] =  df.apply(lambda x: IoU(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)

print('BERT + CLS random')

print(df['Precision'].mean())
print(df['Recall'].mean())
print(df['F1'].mean())
print(df['IoU'].mean())

# %%
df[['Norm_bounds', 'Norm_rep']] =  df.apply(lambda x: normalize(x['Bounds_predetti'], x['Repertori_predetti']), axis=1)

df['Norm_precision'] =  df.apply(lambda x: precision(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_recall'] =  df.apply(lambda x: recall(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)
df['Norm_f1'] =  df.apply(lambda x: f1(x['Norm_precision'], x['Norm_recall']), axis=1)
df['Norm_IoU'] =  df.apply(lambda x: IoU(x['Norm_bounds'], x['Bounds'], x['Norm_rep'], x['Repertori']), axis=1)

print('BERT + CLS random norm')
print(df['Norm_precision'].mean())
print(df['Norm_recall'].mean())
print(df['Norm_f1'].mean())
print(df['Norm_IoU'].mean())


