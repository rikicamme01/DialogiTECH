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

seed_everything(4321)

#cluster paths
df = pd.read_csv('./RepML/data/Splitted_full/Hyperion_train.csv', na_filter=False)
test_df = pd.read_csv('./RepML/data/Splitted_full/Hyperion_test.csv', na_filter=False)

#local paths
#df = pd.read_csv('./data/Splitted_full/Hyperion_train.csv', na_filter=False)
#test_df = pd.read_csv('./data/Splitted_full/Hyperion_test.csv', na_filter=False)

model_name = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

train_dataset, val_dataset = train_val_split(df, model_name, subsample=False)
test_dataset = HyperionDataset(test_df, model_name)

#class_weights = [107.0, 11.88888888888889, 1.2829003711104767, 8.794520547945206, 17.976,
#    3.984042553191489, 8.881422924901186, 1.1549730146491903, 8.743190661478598, 5.268464243845252,
#    2.036248300860897, 12.803418803418804,7.953982300884956, 4.832258064516129, 4.192164179104478,
#    9.115618661257606, 2.9295958279009127, 4.816720257234727, 21.921951219512195, 5.1068181818181815,
#    1.0, 4.026881720430108, 2.646643109540636]
class_weights = [9.999999999999998, 1.9245283018867925, 1.0240198428301348, 1.6617989144481777, 2.4413584905660377,
  1.2533621035728624, 1.6691774181519874, 1.0131580861494596, 1.6574407165406357, 1.3624167754208232,
  1.08798334629951, 2.002177068214804, 1.5904324595091, 1.3253804017041997, 1.271032807659814,
  1.689061961804891, 1.1638336080293228, 1.324061153916156, 2.7763920846755634, 1.3486921097770155,
  1.0, 1.2569993913572732, 1.1398093206213749]
#class_weights = [29.04140787,  3.2268231,   0.34819844,  2.38696503,  4.87895652,  1.08132902,
#  2.41055164,  0.31347703,  2.37303333,  1.42994036,  0.55266839,  3.47504026,
#  2.15883032,  1.31154745,  1.13781635,  2.47411588,  0.79513633,  1.30733026,
#  5.94994698,  1.38606719,  0.27141503,  1.09295621,  0.71833871]
  
trainer = MPTrainer(batch_size, learning_rate, n_epochs, torch.nn.NLLLoss(weight = torch.Tensor(class_weights)))


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=23)
model.name = 'AlBERTo'


trainer.fit(model,train_dataset, val_dataset)
trainer.test(model,test_dataset)
