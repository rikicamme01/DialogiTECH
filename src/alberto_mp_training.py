import pandas as pd
import neptune.new as neptune
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn import preprocessing
import time
import datetime
import  torchmetrics
from torch.utils.data import DataLoader
from transformers import  AdamW
import numpy as np
import sys
import random
import os

if len(sys.argv) != 3:
    print("Usage:", sys.argv[0], "batch_size n_epochs")
    sys.exit(1)

# Hyperparameters
learning_rate = 1e-5
batch_size = int(sys.argv[1])
n_epochs = int(sys.argv[2])

# Test CUDA
print(torch.rand(1, device="cuda"))


#Neptune initialization
run = neptune.init(
    project="mibo8/Rep",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZmRkYThiZi1mZGNlLTRlMTktODQwNS1hNWFlMWQ2Mjc4N2IifQ==",
)


#Deterministic mode
def seed_everything(seed=1464):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# Dataset loading and preprocessing
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

df = pd.read_csv('./RepML/data/Original_csv/Hyperion.csv', na_filter=False)
df = fill_null_features(df)
filter = df["Repertorio"] != ""
df = df[filter]

# lower entire dataset
df = df.applymap(str.lower)

#Lables uniformation uncased
df['Repertorio'].replace('implicazioni','implicazione', inplace=True)
df['Repertorio'].replace('previsioni','previsione', inplace=True)

# split
test_set_perc = 0.2
val_set_perc = 0.1

gb = df.groupby('Repertorio')
train_list = []
test_list = []
val_list = []

for x in gb.groups:
    class_df = gb.get_group(x)

    # Test set creation
    test = class_df.sample(frac=test_set_perc, random_state=1464)
    train = pd.concat([class_df,test]).drop_duplicates(keep=False)

    # Validation set creation
    val = train.sample(frac=val_set_perc)
    train = pd.concat([train,val]).drop_duplicates(keep=False)

    train_list.append(train)
    test_list.append(test)
    val_list.append(val)

train_df = pd.concat(train_list)
test_df = pd.concat(test_list)
val_df = pd.concat(val_list)

dataset_info = {
    'training_set_size' : 1 - test_set_perc,
    'validation_set_size' : val_set_perc,
    'test_set_size' : test_set_perc
}
run['dataset'] = dataset_info

# dataset subclass definition
class HyperionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

#Import model 
tokenizer = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
model = AutoModelForSequenceClassification.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0", num_labels=23)
run['model'] = "bert-base-multilingual-uncased"

#Dataset  creation
training_encodings = tokenizer(
            train_df['Stralcio'].tolist(),
            train_df['Domanda'].tolist(),
            max_length=512,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            truncation=True
        )
test_encodings = tokenizer(
            test_df['Stralcio'].tolist(),
            test_df['Domanda'].tolist(),
            max_length=512,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            truncation=True
)
val_encodings = tokenizer(
            val_df['Stralcio'].tolist(),
            #val_df['Domanda'].tolist(),
            max_length=512,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            truncation=True
)


labels = [
    'anticipazione',
    'causa',
    'commento',
    'conferma',
    'considerazione',
    'contrapposizione',
    'deresponsabilizzazione',
    'descrizione',
    'dichiarazione di intenti',
    'generalizzazione',
    'giudizio',
    'giustificazione',
    'implicazione',
    'non risposta',
    'opinione',
    'possibilità',
    'prescrizione',
    'previsione',
    'proposta',
    'ridimensionamento',
    'sancire',
    'specificazione',
    'valutazione']

le = preprocessing.LabelEncoder()
le.fit(labels)

train_dataset = HyperionDataset(training_encodings,le.transform(train_df['Repertorio']))
test_dataset = HyperionDataset(test_encodings,le.transform(test_df['Repertorio']))
val_dataset = HyperionDataset(val_encodings,le.transform(val_df['Repertorio']))


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Metrics initialization
metric_collection = torchmetrics.MetricCollection({

    'accuracy_micro' : torchmetrics.Accuracy(num_classes=23, multiclass=True, average='micro'),
    'accuracy_macro' : torchmetrics.Accuracy(num_classes=23, multiclass=True, average='macro'),
    'accuracy_weighted' : torchmetrics.Accuracy(num_classes=23, multiclass=True, average='weighted'),
    'accuracy_none' : torchmetrics.Accuracy(num_classes=23, multiclass=True, average='none'),

    'f1_micro' : torchmetrics.F1(num_classes=23, multiclass=True, average='micro'),
    'f1_macro' : torchmetrics.F1(num_classes=23, multiclass=True, average='macro'),
    'f1_weighted' : torchmetrics.F1(num_classes=23, multiclass=True, average='weighted'),
    'f1_none' : torchmetrics.F1(num_classes=23, multiclass=True, average='none'),

    'precision_micro' : torchmetrics.Precision(num_classes=23, multiclass=True, average='micro'),
    'precision_macro' : torchmetrics.Precision(num_classes=23, multiclass=True, average='macro'),
    'precision_weighted' : torchmetrics.Precision(num_classes=23, multiclass=True, average='weighted'),
    'precision_none' : torchmetrics.Precision(num_classes=23, multiclass=True, average='none'),

    'recall_micro' : torchmetrics.Recall(num_classes=23, multiclass=True, average='micro'),
    'recall_macro' : torchmetrics.Recall(num_classes=23, multiclass=True, average='macro'),
    'recall_weighted' : torchmetrics.Recall(num_classes=23, multiclass=True, average='weighted'),
    'recall_none' : torchmetrics.Recall(num_classes=23, multiclass=True, average='none')
})



params_info = {
    'learning_rate' : learning_rate,
    'batch_size' : batch_size,
    'n_epochs' : n_epochs
}
run['params'] = params_info

torch.cuda.empty_cache()
#----------TRAINING

# Measure the total training time for the whole run.
total_t0 = time.time()

epochs = n_epochs

# Creation of Pytorch DataLoaders with shuffle=True for the traing phase
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#Adam algorithm optimized for tranfor architectures
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Scaler for mixed precision
scaler = torch.cuda.amp.GradScaler()

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

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode: Dropout layers are active
    model.train()
    
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 10 == 0 and not step == 0:
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


        # Perform a forward pass in mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(b_input_ids, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
        
        loss = outputs[0]
        logits = outputs[1]

        


        

        # Move logits and labels to CPU
        logits = logits.detach().cpu()
        label_ids = b_labels.to('cpu')

        batch_metric = metric_collection.update(logits, label_ids)
        #print(batch_metric)

        # Perform a backward pass to compute the gradients in MIXED precision
        scaler.scale(loss).backward()

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end.
        total_train_loss += loss.item()

        # Unscales the gradients of optimizer's assigned params in-place before the gradient clipping
        scaler.unscale_(optimizer)

        # Clip the norm of the gradients to 1.0.
        # This helps and prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient in MIXED precision
        scaler.step(optimizer)
        scaler.update()


    # Compute the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    final_metrics = metric_collection.compute()
    print(final_metrics)
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.3f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure performance on
    # the validation set.

    print("")
    print("Running Validation...")

    metric_collection.reset()
    t0 = time.time()

    # Put the model in evaluation mode: the dropout layers behave differently
    model.eval()

    total_val_loss = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # copy each tensor to the GPU using the 'to()' method
        #
        # 'batch' contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for training.
        with torch.no_grad():        

            # Forward pass, calculate logits
            # argmax(logits) = argmax(Softmax(logits))
            outputs = model(b_input_ids, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            
        # Accumulate the validation loss.
        total_val_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu()
        label_ids = b_labels.to('cpu')

        # metric on current batch
        batch_metric = metric_collection.update(logits.softmax(dim=1), label_ids)

    # Report the final metrics for this validation phase.
    # metric on all batches using custom accumulation from torchmetrics library

    final_metrics = metric_collection.compute()
    print('VALIDATION: ')
    print(final_metrics)
    # Compute the average loss over all of the batches.
    avg_val_loss = total_val_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))


print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



# ========================================
#               Test
# ========================================
# Measure performance on
# the validation set.

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("")
print("Running Test...")

metric_collection.reset()
t0 = time.time()

# Put the model in evaluation mode: the dropout layers behave differently
model.eval()

total_test_loss = 0

# Evaluate data for one epoch
for batch in test_dataloader:
    
    # Unpack this training batch from our dataloader. 
    #
    # copy each tensor to the GPU using the 'to()' method
    #
    # 'batch' contains three pytorch tensors:
    #   [0]: input ids 
    #   [1]: attention masks
    #   [2]: labels 
    b_input_ids = batch['input_ids'].to(device)
    b_input_mask = batch['attention_mask'].to(device)
    b_labels = batch['labels'].to(device)
    
    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for training.
    with torch.no_grad():        

        # Forward pass, calculate logits
        # argmax(logits) = argmax(Softmax(logits))
        outputs = model(b_input_ids, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        
    # Accumulate the validation loss.
    total_test_loss += loss.item()

    # Move logits and labels to CPU
    logits = logits.detach().cpu()
    label_ids = b_labels.to('cpu')

    # metric on current batch

    batch_metric = metric_collection.update(logits.softmax(dim=1), label_ids)

# Report the final metrics for this validation phase.
# metric on all batches using custom accumulation from torchmetrics library

test_metrics = metric_collection.compute()
print(' Test metrics: ')
print(final_metrics)

run['metrics'] = final_metrics
# Compute the average loss over all of the batches.
avg_test_loss = total_test_loss / len(test_dataloader)

run['test/loss'] = avg_test_loss

# Measure how long the validation run took.
test_time = format_time(time.time() - t0)

print("  Test Loss: {0:.2f}".format(avg_test_loss))
print("  Test took: {:}".format(test_time))

#torch.save(model.state_dict(), './')

run.stop()