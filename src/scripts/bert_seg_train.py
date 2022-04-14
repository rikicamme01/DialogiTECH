import yaml
import os
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

import pandas as pd
import torch
from transformers import BertForTokenClassification
from ast import literal_eval
import neptune.new as neptune

from utils.utils import plot_loss, seed_everything
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

logger = NeptuneLogger()
logger.run['config'] = config

df = pd.read_csv(sys.argv[1] + 'data/processed/splitted_union/ie_hyperion_train.csv', converters={'Stralci': literal_eval, 'Repertori': literal_eval})
test_df = pd.read_csv(sys.argv[1] + 'data/processed/splitted_union/ie_s2_hyperion_test.csv', converters={'Stralci': literal_eval, 'Repertori': literal_eval})

model_name = config['model']

model = BertForTokenClassification.from_pretrained(
    model_name, num_labels=2)

model.name = model_name
train_dataset, val_dataset = train_val_split(df, model_name)
test_dataset = IEHyperionDataset(test_df, model_name)

trainer = BertSegTrainer()

history = trainer.fit(model,
            train_dataset, 
            val_dataset,
            config['batch_size'],
            float(config['learning_rate']),
            config['n_epochs'],
            torch.nn.CrossEntropyLoss(weight = torch.Tensor(config['loss_weights'])))

logger.run['history'] = history
fig = plot_loss(history['train_loss'], history['val_loss'])
logger.run["loss_plot"].upload(neptune.types.File.as_image(fig))

out = trainer.test(model, val_dataset, config['batch_size'], torch.nn.CrossEntropyLoss(weight = torch.Tensor(config['loss_weights'])))
logger.run['val/norm_metrics'] = out['normalized_metrics']
logger.run['val/metrics'] = out['metrics']
logger.run['val/loss'] = out['loss']
logger.run['val/predicted_spans'] = out['predicted_spans']


out = trainer.test(model, test_dataset, config['batch_size'], torch.nn.CrossEntropyLoss(weight = torch.Tensor(config['loss_weights'])))
logger.run['test/norm_metrics'] = out['normalized_metrics']
logger.run['test/metrics'] = out['metrics']
logger.run['test/loss'] = out['loss']
logger.run['test/predicted_spans'] = out['predicted_spans']
