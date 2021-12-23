import pandas as pd
from transformers import AutoModelForSequenceClassification

from datasets.hyperion_dataset import train_val_split
from trainers.mp_trainer import MPTrainer

df = pd.read_csv('../data/Original_csv/Hyperion.csv', na_filter=False)

tok_name = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

train_dataset, val_dataset = train_val_split(df.head(100), tok_name)

trainer = MPTrainer(2, 1e-5, 1)


model = AutoModelForSequenceClassification.from_pretrained(tok_name, num_labels=23)

trainer.fit(model,train_dataset, val_dataset)
