import yaml
import os
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification

from utils.utils import seed_everything
from loggers.neptune_logger import NeptuneLogger
from trainers.bert_seg_trainer import BertSegTrainer
from datasets.ie_hyperion_dataset import IEHyperionDataset, train_val_split

if len(sys.argv) != 2:
    print("ERROR:  config_file path not provided")
    sys.exit(1)

# Repository paths
# './' local
# './RepML/ cluster

try: 
    with open (sys.argv[1] + 'config/bert_seg_train.yml', 'r') as file:
        config = yaml.safe_load(file)        
except Exception as e:
    print('Error reading the config file')
    sys.exit(1)
print('config file loaded!')

seed_everything(config['seed'])

#logger = NeptuneLogger()
#logger.run['config'] = config

df = pd.read_csv(sys.argv[1] + 'data/processed/splitted_union/ie_hyperion_train.csv')
test_df = pd.read_csv(sys.argv[1] + 'data/processed/splitted_union/ie_hyperion_test.csv')

model_name = config['model']

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=2)
model.name = model_name
train_dataset, val_dataset = train_val_split(df.head(20), model_name)
test_dataset = IEHyperionDataset(test_df.head(20), model_name)

trainer = BertSegTrainer()

history = trainer.fit(model,
            train_dataset, 
            val_dataset,
            config['batch_size'],
            float(config['learning_rate']),
            config['n_epochs'],
            torch.nn.CrossEntropyLoss(weight = torch.Tensor(config['loss_weights'])))

#logger.run['history'] = history