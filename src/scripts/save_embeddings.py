import sys
import os 
sys.path.append(os.path.dirname(sys.path[0]))
import pandas as pd

train_df = pd.read_csv('./data/raw/sentipolc/training_set_sentipolc16.csv')
test_df = pd.read_csv('./data/raw/sentipolc/test_set_sentipolc16_gold2000.csv', escapechar='\\')

from models.bert_rep import BertRep
import numpy as np
import torch

bert_rep = BertRep()
X = train_df.apply(lambda x: bert_rep.hidden_states([x['text']]), axis=1).to_list()
X = torch.stack(X)
torch.save(X, './data/other/embeddings/train_embeddings.pt')

test_df['hs'] = test_df.apply(lambda x: bert_rep.hidden_states([x['text']]), axis=1).to_list()
X = torch.stack(X)
torch.save(X, './data/other/embeddings/test_embeddings.pt')