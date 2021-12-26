import pandas as pd
from transformers import AutoModelForSequenceClassification
from datasets import HyperionDataset

from datasets.hyperion_dataset import train_val_split
from trainers.mp_trainer import MPTrainer

df = pd.read_csv('../data/Splitted_full/Hyperion_train.csv', na_filter=False)
test_df = pd.read_csv('../data/Splitted_full/Hyperion_test.csv', na_filter=False)

model_name = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

train_dataset, val_dataset = train_val_split(df.head(100), model_name)
test_dataset = HyperionDataset(test_df.head(200), model_name)

trainer = MPTrainer(2, 1e-5, 2)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=23)

trainer.fit(model,train_dataset, val_dataset)
trainer.test(model,test_dataset)
