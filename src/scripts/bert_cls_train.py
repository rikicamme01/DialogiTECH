#%%
import yaml
import os
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

import pandas as pd
import torch
import neptune

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets.hyperion_dataset import HyperionDataset
from datasets.hyperion_dataset import train_val_split
from trainers.bert_cls_trainer import BertClsTrainer
from utils.utils import seed_everything
from loggers.neptune_logger import NeptuneLogger
from utils.utils import plot_confusion_matrix, plot_f1, plot_loss

if len(sys.argv) != 2:
    print("ERROR:  config_file path not provided")
    sys.exit(1)

# Repository paths
# './' local
# './RepML/ cluster

try: 
    with open (sys.argv[1] + 'config/bert_cls_train.yml', 'r') as file:
        config = yaml.safe_load(file)        
except Exception as e:
    print('Error reading the config file')
    sys.exit(1)
print('config file loaded!')
#%%
seed_everything(config['seed'])

df = pd.read_csv(sys.argv[1] + 'data/processed/splitted_full/hyperion_train.csv', na_filter=False)
test_df = pd.read_csv(sys.argv[1] + 'data/processed/splitted_full/hyperion_test.csv', na_filter=False)

logger = NeptuneLogger()
logger.run['config'] = config


model_name = config['model'] # runno una volta per copiarmelo sul mio hugging e dopo modifico file config impostando il mio "nuovo" come modello predefinito su
                            # su cui allenare

train_dataset, val_dataset = train_val_split(df, model_name, subsample=False)
test_dataset = HyperionDataset(test_df, model_name)

trainer = BertClsTrainer()


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=23) #
model.name = model_name

history = trainer.fit(model,
            train_dataset, 
            val_dataset,
            config['batch_size'],
            float(config['learning_rate']),
            config['n_epochs'],
            torch.nn.CrossEntropyLoss(weight = torch.Tensor(config['loss_weights'])))

logger.run['history'] = history

out = trainer.test(model,test_dataset, config['batch_size'], torch.nn.CrossEntropyLoss(weight = torch.Tensor(config['loss_weights'])))
logger.run['test/metrics'] = out['metrics']
logger.run['test/loss'] = out['loss']

cm = plot_confusion_matrix(out['gt'], out['pred'], test_dataset.labels_list())
logger.run["confusion_matrix"].upload(neptune.types.File.as_image(cm))

fig = plot_loss(history['train_loss'], history['val_loss'])
logger.run["loss_plot"].upload(neptune.types.File.as_image(fig))


#%%
#hf_token = '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' # con nuovo profilo
hf_token = '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'
if config['save']:
    model.push_to_hub("BERT_DialogicaPD", use_temp_dir=True, use_auth_token=hf_token)
    AutoTokenizer.from_pretrained(model_name).push_to_hub("BERT_DialogicaPD", use_temp_dir=True, use_auth_token=hf_token)