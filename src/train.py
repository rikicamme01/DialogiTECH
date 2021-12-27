import sys

import pandas as pd
import torch
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
#df = pd.read_csv('./data/Splitted_full/Hyperion_train.csv', na_filter=False)
#test_df = pd.read_csv('./data/Splitted_full/Hyperion_test.csv', na_filter=False)

model_name = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

train_dataset, val_dataset = train_val_split(df, model_name, subsample=False)
test_dataset = HyperionDataset(test_df, model_name)

class_weights = [107.0, 11.88888888888889, 1.2829003711104767, 8.794520547945206, 17.976,
    3.984042553191489, 8.881422924901186, 1.1549730146491903, 8.743190661478598, 5.268464243845252,
    2.036248300860897, 12.803418803418804,7.953982300884956, 4.832258064516129, 4.192164179104478,
    9.115618661257606, 2.9295958279009127, 4.816720257234727, 21.921951219512195, 5.1068181818181815,
    1.0, 4.026881720430108, 2.646643109540636]
trainer = MPTrainer(batch_size, learning_rate, n_epochs, torch.nn.NLLLoss(weight = torch.HalfTensor(class_weights)))


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=23)
model.name = 'AlBERTo'


trainer.fit(model,train_dataset, val_dataset)
trainer.test(model,test_dataset)
