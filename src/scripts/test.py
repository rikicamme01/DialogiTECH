import sys
import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets.hyperion_dataset import HyperionDataset
from trainers.mp_trainer import MPTrainer
from utils.utils import seed_everything
from datasets.hyperion_dataset import decode_labels
import neptune.new as neptune

batch_size = 2

seed_everything(4321)

#cluster paths
test_df = pd.read_csv('./RepML/data/Splitted_full/Hyperion_test.csv', na_filter=False)

#local paths
#test_df = pd.read_csv('./data/Splitted_full/Hyperion_test.csv', na_filter=False)

model_name = "dbmdz/bert-base-italian-xxl-uncased"
test_dataset = HyperionDataset(test_df, model_name)

learning_rate = 1e-5
batch_size = 2
n_epochs = 1

trainer = MPTrainer(batch_size, learning_rate, n_epochs, torch.nn.NLLLoss())

model = AutoModelForSequenceClassification.from_pretrained('MiBo/RepML')

pred, gt = trainer.test(model,test_dataset)
error_dict = {
    'Stralcio' : test_df['Stralcio'].tolist(),
    'Repertorio_corretto' : decode_labels(gt),
    'Repertorio_predetto' : decode_labels(pred)
}

error_df = pd.DataFrame.from_dict(error_dict)
error_df.drop(error_df[error_df.Repertorio_corretto == error_df.Repertorio_predetto].index, inplace=True)
error_df.reset_index(drop=True)
error_df.to_csv('error.csv', index = False, header=True)
trainer.logger.run["error_csv"].upload('error.csv')
