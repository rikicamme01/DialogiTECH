#!/usr/bin/env python3
"""
# Import

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn import preprocessing
import time
import datetime
from torch.utils.data import DataLoader
from transformers import  AdamW
import  torchmetrics
import numpy as np

# Utility functions

def fill_null_features(df):
    for c in ['Domanda','Testo']:
        for i in range(0,len(df.index)):  
            if not df[c][i]:
                j=i
                while j>0: 
                    j-=1
                    if df[c][j]:
                        df[c][i] = df[c][j]
                        break
    return df


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Loading and preprocessing dataset

df = pd.read_csv('../data/Original_csv/Hyperion.csv', na_filter=False)
df = fill_null_features(df)
filter = df["Stralcio"] != ""
df = df[filter]
filter = df["Repertorio"] != ""
df = df[filter]

#Lables uniformation

df['Repertorio'].replace('Implicazioni','Implicazione', inplace=True)
df['Repertorio'].replace('Previsioni','Previsione', inplace=True)
df['Repertorio'].replace('causa','Causa', inplace=True)
df['Repertorio'].replace('commento','Commento', inplace=True)
df['Repertorio'].replace('contrapposizione','Contrapposizione', inplace=True)
df['Repertorio'].replace('generalizzazione','Generalizzazione', inplace=True)
df['Repertorio'].replace('giudizio','Giudizio', inplace=True)
df['Repertorio'].replace('prescrizione','Prescrizione', inplace=True)
df['Repertorio'].replace('previsione','Previsione', inplace=True)
df['Repertorio'].replace('sancire','Sancire', inplace=True)
df['Repertorio'].replace('specificazione','Specificazione', inplace=True)
df['Repertorio'].replace('valutazione','Valutazione', inplace=True)

# loading pretrained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=23)

#Torch dataset
X_encodings = tokenizer(
            df['Stralcio'].tolist()[:1000],
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            truncation=True
        )

labels = [
    'Anticipazione',
    'Causa',
    'Commento',
    'Conferma',
    'Considerazione',
    'Contrapposizione',
    'Deresponsabilizzazione',
    'Descrizione',
    'Dichiarazione di intenti',
    'Generalizzazione',
    'Giudizio',
    'Giustificazione',
    'Implicazione',
    'Non risposta',
    'Opinione',
    'Possibilità',
    'Prescrizione',
    'Previsione',
    'Proposta',
    'Ridimensionamento',
    'Sancire',
    'Specificazione',
    'Valutazione']

le = preprocessing.LabelEncoder()
le.fit(labels)
dataset = HyperionDataset(X_encodings,le.transform(df['Repertorio'][:1000]))

train_dataset_size = int(len(dataset) * 0.7)
val_dataset_size = int(len(dataset) * 0.1)
test_dataset_size = len(dataset) - train_dataset_size - val_dataset_size # 0.2
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_dataset_size, val_dataset_size, test_dataset_size])

# Traning phase
# Measure the total training time for the whole run.
total_t0 = time.time()

epochs = 1

# Creation of Pytorch DataLoaders with shuffle=True for the traing phase
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

#Adam algorithm optimized for tranfor architectures
optimizer = AdamW(model.parameters(), lr=1e-5)

# Setup for training with gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # initialize metric
    metric = torchmetrics.Accuracy(num_classes=23)

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode: Dropout layers are active
    model.train()
    
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Compute time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from the dataloader. 
        #
        #  copy each tensor to the GPU using the 'to()' method
        #
        # 'batch' contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        # clear any previously calculated gradients before performing a
        # backward pass
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        outputs = model(b_input_ids, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
        
        loss = outputs[0]
        logits = outputs[1]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end.
        total_train_loss += loss.item()

        # Perform a backward pass to compute the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This helps and prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient
        optimizer.step()


    # Compute the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.3f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

# Save model
torch.save(model.state_dict(), 'bert.pt')
"""
print('hello world')
