import yaml
import os
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

import pandas as pd
import torch
import neptune.new as neptune

from transformers import AutoModelForSequenceClassification
from datasets.hyperion_dataset import HyperionDataset
from datasets.hyperion_dataset import train_val_split
from trainers.mp_trainer import MPTrainer
from utils.utils import seed_everything
from loggers.neptune_logger import NeptuneLogger
from utils.utils import plot_confusion_matrix, plot_f1, plot_loss



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

logger = NeptuneLogger()
logger.run['config'] = config


model_name = config['model']

train_dataset, val_dataset = train_val_split(df, model_name, subsample=False)
test_dataset = HyperionDataset(test_df, model_name)

trainer = MPTrainer()


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=23)
model.name = model_name
"""
history = trainer.fit(model,
            train_dataset, 
            val_dataset,
            config['batch_size'],
            float(config['learning_rate']),
            config['n_epochs'],
            torch.nn.NLLLoss(weight = torch.Tensor(config['loss_weights'])))

logger.run['history'] = history
"""
out = trainer.test(model,test_dataset)
logger.run['test/metrics'] = out['metrics']
logger.run['test/loss'] = out['loss']

cm = plot_confusion_matrix(out['gt'], out['pred'], test_dataset.labels_list())
logger.run["confusion_matrix"].upload(neptune.types.File.as_image(cm))

hf_token = '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'
if config['save']:
    model.push_to_hub("RepML", use_temp_dir=True, use_auth_token=hf_token)