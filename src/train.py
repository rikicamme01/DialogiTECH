import sys

import pandas as pd
from transformers import AutoModelForSequenceClassification
from datasets.hyperion_dataset import HyperionDataset

from datasets.hyperion_dataset import train_val_split
from trainers.mp_trainer import MPTrainer
from utils.utils import seed_everything

if len(sys.argv) != 3:
    print("ERROR:  batch_size n_epochs  not provided")
    sys.exit(1)

# Hyperparameters
learning_rate = 1e-5
batch_size = int(sys.argv[1])
n_epochs = int(sys.argv[2])

seed_everything(1464)

#cluster paths
df = pd.read_csv('./RepML/data/Splitted_full/Hyperion_train.csv', na_filter=False)
test_df = pd.read_csv('./RepML/data/Splitted_full/Hyperion_test.csv', na_filter=False)

#local paths
#df = pd.read_csv('../data/Splitted_full/Hyperion_train.csv', na_filter=False)
#test_df = pd.read_csv('../data/Splitted_full/Hyperion_test.csv', na_filter=False)

model_name = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

train_dataset, val_dataset = train_val_split(df, model_name, subsample=True)
test_dataset = HyperionDataset(test_df, model_name)

trainer = MPTrainer(batch_size, learning_rate, n_epochs)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=23)
model.name = 'AlBERTo'


trainer.fit(model,train_dataset, val_dataset)
trainer.test(model,test_dataset)
