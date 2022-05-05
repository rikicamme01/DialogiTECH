import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from ast import literal_eval
from loggers.neptune_logger import NeptuneLogger
from src.datasets.ie_hyperion_dataset import find_word_bounds
from src.models.nltk_segmenter import NLTKSegmenter
from src.models.bert_rep import BertRep

if len(sys.argv) != 2:
    print("ERROR:  config_file path not provided")
    sys.exit(1)

logger = NeptuneLogger()

df = pd.read_csv(sys.argv[1] + 'data/processed/pipeline/test/ie_hyperion.csv', converters={'Stralci': literal_eval, 'Repertori': literal_eval})
df['Bounds'] = df.apply(lambda x: find_word_bounds(x['Stralci'], x['Testo']), axis=1).values.tolist()

nltk_seg = NLTKSegmenter()
df['Stralci_predetti'] = df['Testo'].map(nltk_seg.predict).values.tolist()
df['Bounds_predetti'] = df.apply(lambda x: find_word_bounds(x['Stralci_predetti'], x['Testo']), axis=1).values.tolist()

bert_rep = BertRep()
df['Repertori_predetti'] = df['Stralci_predetti'].map(bert_rep.predict).values.tolist()

def intersection(A, B):
    if A == B:
        return 1
    start = max(A[0], B[0])
    end = min(A[1], B[1])
    if(start > end):
        return 0
    return end - start + 1

def C(pred_bound:tuple, gt_bound:tuple, pred_rep:str, gt_rep:str, norm_factor:int) -> float:
    if pred_rep != gt_rep:
        return 0
    return intersection(pred_bound, gt_bound) / norm_factor

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


df['Precision'] =  df.apply(lambda x: precision(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['Recall'] =  df.apply(lambda x: recall(x['Bounds_predetti'], x['Bounds'], x['Repertori_predetti'], x['Repertori']), axis=1)
df['F1'] =  df.apply(lambda x: f1(x['Precision'], x['Recall']), axis=1)

logger.run['pipeline'] = 'NLTK + BERT (No normalization)'
logger.run['precision'] = df['Precision'].mean()
logger.run['recall'] = df['recall'].mean()
logger.run['f1'] = df['F1'].mean()



