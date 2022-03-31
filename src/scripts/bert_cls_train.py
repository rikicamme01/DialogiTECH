import yaml
import os
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

import pandas as pd
import torch

from transformers import AutoModelForSequenceClassification
from datasets.hyperion_dataset import HyperionDataset
from datasets.hyperion_dataset import train_val_split
from trainers.mp_trainer import MPTrainer
from utils.utils import seed_everything



try: 
    with open ('./config/bert_cls_train.yml', 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config file')

seed_everything(config['seed'])

if config['mode'] == 'cluster':
    df = pd.read_csv('./RepML/data/processed/Splitted_full/Hyperion_train.csv', na_filter=False)
    test_df = pd.read_csv('./RepML/data/processed/Splitted_full/Hyperion_test.csv', na_filter=False)
elif config['mode'] == 'sagemaker':
    df = pd.read_csv('../data/processed/Splitted_full/Hyperion_train.csv', na_filter=False)
    test_df = pd.read_csv('../data/processed/Splitted_full/Hyperion_test.csv', na_filter=False)
else:
    df = pd.read_csv('./data/processed/Splitted_full/Hyperion_train.csv', na_filter=False)
    test_df = pd.read_csv('./data/processed/Splitted_full/Hyperion_test.csv', na_filter=False)

model_name = config['model']

train_dataset, val_dataset = train_val_split(df, model_name, subsample=False)
test_dataset = HyperionDataset(test_df, model_name)

trainer = MPTrainer(config['batch_size'],
            float(config['learning_rate']),
            config['n_epochs'],
            torch.nn.NLLLoss(weight = torch.Tensor(config['loss_weights'])))


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=23)
model.name = model_name

trainer.fit(model,train_dataset, val_dataset)
trainer.test(model,test_dataset)

hf_token = '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'
if config['save']:
    model.push_to_hub("RepML", use_temp_dir=True, use_auth_token=hf_token)