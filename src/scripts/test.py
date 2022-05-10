import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from ast import literal_eval
from loggers.neptune_logger import NeptuneLogger
from datasets.ie_hyperion_dataset import find_word_bounds
from models.bert_segmenter import BertSegmenter
from models.bert_rep import BertRep

df = pd.read_csv('./data/processed/pipeline/test/ie_hyperion.csv', converters={'Stralci': literal_eval, 'Repertori': literal_eval})
df['Bounds'] = df.apply(lambda x: find_word_bounds(x['Stralci'], x['Testo']), axis=1).values.tolist()

bert_seg = BertSegmenter()
df['Stralci_predetti'] = df['Testo'].map(bert_seg.predict).values.tolist()
df['Bounds_predetti'] = df.apply(lambda x: find_word_bounds(x['Stralci_predetti'], x['Testo']), axis=1).values.tolist()
