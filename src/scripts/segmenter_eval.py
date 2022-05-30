import os
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

import pandas as pd
import torch
from transformers import BertForTokenClassification
from ast import literal_eval
import neptune.new as neptune
from transformers import AutoTokenizer

from utils.utils import plot_loss, seed_everything
from loggers.neptune_logger import NeptuneLogger
from trainers.bert_seg_trainer import BertSegTrainer, normalize_bounds_by_repertoire
from datasets.ie_hyperion_dataset import IEHyperionDataset, find_word_bounds, train_val_split
from models.nltk_segmenter import NLTKSegmenter

if len(sys.argv) != 2:
    print("ERROR:  config_file path not provided")
    sys.exit(1)

# Repository paths
# './' local
# './RepML/ cluster


seed_everything(4321)

logger = NeptuneLogger()

df = pd.read_csv(sys.argv[1] + 'data/processed/splitted_union/ie_s2_hyperion_test.csv', converters={'Stralci': literal_eval, 'Repertori': literal_eval})


nltk_seg = NLTKSegmenter()
df['Stralci_predetti'] = df['Testo'].map(nltk_seg.predict).values.tolist()
df['Bounds_predetti'] = df.apply(lambda x: find_word_bounds(x['Stralci_predetti'], x['Testo']), axis=1).values.tolist()
df['Bounds_predetti_norm'] = df.apply(lambda x: normalize_bounds_by_repertoire(x['Bounds_predetti'], x))

logger.run['test/metrics'] = compute_metrics(pred_word_bounds, test_dataset)
logger.run['test/norm_metrics'] = compute_metrics(norm_pred_word_bounds, test_dataset)
