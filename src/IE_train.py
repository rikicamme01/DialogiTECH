import pandas as pd
import string
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import numpy as np
import torch
from transformers import AutoModelForTokenClassification
import random
import os
import sys

import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import neptune.new as neptune

import time
import datetime
from torch.nn import utils

import torchmetrics
import torch
from torch.utils.data import DataLoader
from transformers import  AdamW
from transformers import get_constant_schedule_with_warmup
import numpy as np

if len(sys.argv) != 4:
    print("ERROR:  batch_size n_epochs model_name  not provided")
    sys.exit(1)

# Hyperparameters
learning_rate = 1e-5
batch_size = int(sys.argv[1])
n_epochs = int(sys.argv[2])
model_name = str(sys.argv[3])



#df = pd.read_csv('./RepML/data/Original_csv/Hyperion.csv')
df = pd.read_csv('./data/Original_csv/Hyperion.csv')

def find_char_bounds(spans: list, text: str) -> list:
    '''
    Given a list of spans and a text, find the start and end indices of each span in the text.
    Indeces are computed counting chars
    
    :param spans: a list of strings to search for
    :type spans: list
    :param text: the text to search
    :type text: str
    :return: A list of tuples, where each tuple contains the start and end index of a span.
    '''
    bounds = []
    last_char = 0
    for span in spans:
        start = text.find(span)
        if start == -1:
            start = last_char + 1
        bounds.append((start, start + len(span)-1))
        last_char = start + len(span)-1
    return bounds


def IE_gen(bounds: list, text:str) -> str:
    tags = ['I' for i in range(len(text))]
    for bound in bounds:
        if bound[1] < len(text):
            tags[bound[1]] = 'E'
        else:
            tags[-1] = 'E'
    return ''.join(tags)


dataset = []

for row in df.itertuples():
    text = row.Testo
    if pd.isna(text):
        sample['Stralci'].append(row.Stralcio)
        sample['Repertori'].append(row.Repertorio)
        
    else: 
        sample = {}
        sample['Testo'] = text
        sample['Stralci'] = [row.Stralcio]
        
        sample['Repertori'] = [row.Repertorio]
        dataset.append(sample)
        
IE_dict = {
    'Testo': [sample['Testo'] for sample in dataset],
    'Tags': [sample['Tags'] for sample in dataset],
    'Bounds': [sample['Bounds'] for sample in dataset],
    'Repertori': [sample['Repertori'] for sample in dataset]
}
IE_df = pd.DataFrame(IE_dict)

LABELS = [
                'I',
                'E'
        ]

def encode_labels(labels_list):
    le = preprocessing.LabelEncoder()
    le.fit(LABELS)
    return le.transform(labels_list)


class IE_Hyperion_dataset(Dataset):
    def __init__(self, df, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.df = df

    def __getitem__(self, idx):
        text = self.df['Testo'].iloc[idx]
        encoding = self.tokenizer(text,
                                  # is_pretokenized=True,
                                  return_special_tokens_mask=True,
                                  return_offsets_mapping=True,
                                  add_special_tokens=True,
                                  return_attention_mask=True,
                                  padding='max_length',
                                  truncation=True,
                                  )
        char_labels = encode_labels(list(self.df['Tags'].iloc[idx]))

        encoded_labels = np.ones(len(encoding['input_ids']), dtype=int) * -100
        for idx, e in enumerate(encoding['offset_mapping']):
            if e[1] != 0:
                # overwrite label
                if 0 in char_labels[e[0]:e[1]]:
                    encoded_labels [idx] = 0
                else:
                    encoded_labels [idx] = 1


        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        return item

    def __len__(self):
        return len(self.df.index)


train_size = 0.8
train_df = IE_df.sample(frac=train_size,random_state=200)
test_df = IE_df.drop(train_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)
val_size = 0.2
val_df = train_df.sample(frac=val_size,random_state=200)
train_df = train_df.drop(val_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)

print("FULL Dataset: {}".format(IE_df.shape))
print("TRAIN Dataset: {}".format(train_df.shape))
print("VALIDATION Dataset: {}".format(val_df.shape))
print("TEST Dataset: {}".format(test_df.shape))

train_dataset = IE_Hyperion_dataset(train_df, model_name)
val_dataset = IE_Hyperion_dataset(val_df, model_name) 
test_dataset = IE_Hyperion_dataset(test_df, model_name)

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)

#Deterministic mode
def seed_everything(seed=1464):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def plot_loss(loss, val_loss):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xticks(range(1,len(loss)+1))
    plt.plot(range(1,len(loss)+1), loss, label='train')
    plt.plot(range(1,len(val_loss)+1), val_loss, label='val')
    plt.title('loss')
    plt.legend()
    #plt.savefig('loss.png')
    return fig

def plot_f1(f1, val_f1):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xticks(range(1,len(f1)+1))
    plt.plot(range(1,len(f1)+1), f1, label='train')
    plt.plot(range(1,len(val_f1)+1), val_f1, label='val')
    plt.title('f1')
    plt.legend()
    #plt.savefig('f1.png')
    return fig

def plot_confusion_matrix(y_true, pred, labels):
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, pred, display_labels=labels, normalize='true', values_format='.2f')
    disp.plot(cmap="Blues", values_format='.2g',xticks_rotation='vertical', ax=ax)
    return disp.figure_


class NeptuneLogger():
    def __init__(self) -> None:
        #Neptune initialization
        self.run = neptune.init(
            project="mibo8/Rep",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZmRkYThiZi1mZGNlLTRlMTktODQwNS1hNWFlMWQ2Mjc4N2IifQ==",
        )


class IE_MPTrainer():
    def __init__(self, batch_size, lr, n_epochs) -> None:
        self.batch_size = batch_size
        self.learning_rate = lr
        self.n_epochs = n_epochs

        self.logger = NeptuneLogger()



        
    def fit(self, model, train_dataset, val_dataset):
        self.logger.run['model'] = model_name
        
        params_info = {
            'learning_rate' : self.learning_rate,
            'batch_size' : self.batch_size,
            'n_epochs' : self.n_epochs
        }
        self.logger.run['params'] = params_info

        torch.cuda.empty_cache()
        #----------TRAINING

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        epochs_train_loss = []
        epochs_val_loss = []


        epochs = self.n_epochs

        # Creation of Pytorch DataLoaders with shuffle=True for the traing phase
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        #Adam algorithm optimized for tranfor architectures
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        #scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=300)

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
                #scheduler.step()

            # Compute the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)
            epochs_train_loss.append(avg_train_loss)
            
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

            print('VALIDATION: ')
            
            # Compute the average loss over all of the batches.
            avg_val_loss = total_val_loss / len(validation_dataloader)
            epochs_val_loss.append(avg_val_loss)
            
            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)
            
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

        loss_fig = plot_loss(epochs_train_loss, epochs_val_loss)

        self.logger.run["loss"].upload(neptune.types.File.as_image(loss_fig))
        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



    
    def test(self, model, test_dataset):
        # ========================================
        #               Test
        # ========================================
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Setup for testing with gpu
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        print("")
        print("Running Test...")
        t0 = time.time()

        # Save prediction for confusion matrix
        pred = []

        model.eval()

        total_test_loss = 0

        # Evaluate data for one epoch
        for batch in test_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            b_special_tokens_mask = batch['special_tokens_mask'].to(device)
            with torch.no_grad():        

                # Forward pass, calculate logits
                # argmax(logits) = argmax(Softmax(logits))
                outputs = model(b_input_ids, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                loss = outputs[0]
                logits = outputs[1]
                
            # Accumulate the test loss.
            total_test_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu() #shape (batch_size, seq_len, num_labels)
            label_ids = b_labels.to('cpu')

            batch_pred = logits.softmax(dim=-1)
            full_pred =  batch_pred.argmax(dim=-1)
            

            for i, sample_pred in enumerate(full_pred):
                active_pred = []
                for j, e in enumerate(b_special_tokens_mask[i]):
                    if(e==0):
                        active_pred.append(int(sample_pred[j]))
                pred.append(active_pred)




        avg_test_loss = total_test_loss / len(test_dataloader)
        #self.logger.run['test/loss'] = avg_test_loss
        test_time = format_time(time.time() - t0)

        
        print("  Test Loss: {0:.2f}".format(avg_test_loss))
        print("  Test took: {:}".format(test_time))

        return pred



trainer = IE_MPTrainer(batch_size, learning_rate, n_epochs)

trainer.fit(model, train_dataset, val_dataset)
pred = trainer.test(model,test_dataset)

def pred_to_bounds(pred: list):
    dataset_bounds = []
    for e in pred:
        start = 0
        end = 0
        bounds = []
        for i, tok_pred in enumerate(e):
            if tok_pred == 0:
                end = i
                bounds.append((start, end))
                start = end + 1
        dataset_bounds.append(bounds)
    return dataset_bounds


bert_pred = pred_to_bounds(pred)
gt_bounds = []
for sample in test_dataset:
    gt_bounds.append(pred_to_bounds([sample['labels']])[0])
test_df['Subword_bounds'] = gt_bounds

def IoU(A, B):
    if A == B:
        return 1
    start = max(A[0], B[0])
    end = min(A[1], B[1])
    if(start > end):
        return 0
    intersection = end - start
    return intersection / (A[1] - A[0] + B[1] - B[0] - intersection)


def compute_IoUs(pred_bounds, gt_spans):
    IoUs = []
    for gt_bounds in gt_spans:
        IoUs.append(IoU(pred_bounds, gt_bounds))
    return IoUs


def normalize(text_spans_dict, gt_spans):
    normalized = []
    for i in range(len(text_spans_dict)):
        #normalized is not empty
        if normalized:
            if normalized[-1]['Repertorio'] == text_spans_dict[i]['Repertorio']:
                new_span = (normalized[-1]['Bounds'][0],
                            text_spans_dict[i]['Bounds'][1])
                new_span_features = {
                    'Bounds': new_span,
                    'IoU': None,
                    'Repertorio': text_spans_dict[i]['Repertorio']
                }
                del normalized[-1]
                normalized.append(new_span_features)
            else:
                normalized.append(text_spans_dict[i])
        else:
            normalized.append(text_spans_dict[i])

    for i in range(len(normalized)):
        normalized[i]['IoU'] = max(compute_IoUs(
            normalized[i]['Bounds'], gt_spans['Bounds']))
    return normalized


metrics = []
normalized_metrics = []
for i, pred_bounds in enumerate(bert_pred):
    text_IoUs = []
    for pred_span in pred_bounds:
        IoUs = compute_IoUs(pred_span, test_df.iloc[i]['Subword_bounds'])
        best = np.argmax(IoUs)
        span_features = {
            'Bounds': pred_span,
            'IoU': IoUs[best],
            'Repertorio': test_df.iloc[i]['Repertori'][best]
        }

        text_IoUs.append(span_features)
    metrics.append(text_IoUs)
    normalized_metrics.append(normalize(text_IoUs, test_df.iloc[i]))


print('----------------------------------------------------------')
print('Risultati labels GT e stralci non uniti')


n_spans = 0
for e in test_df['Subword_bounds']:
    n_spans += len(e)
print('Numero stralci nel dataset:', str(n_spans))

n_spans = 0
for e in metrics:
    n_spans += len(e)
print('Numero stralci predetti:', str(n_spans))

mean = 0
long_spans = 0
min_lenght = 0
perfect_spans = 0
for text in metrics:
    for span in text:
        if span['Bounds'][1] - span['Bounds'][0] >= min_lenght:
            long_spans += 1
            mean += span['IoU']
            if span['IoU'] == 1:
                perfect_spans += 1
perfect_spans_perc = perfect_spans / long_spans
mean_IoU = mean / long_spans
print('Numero stralci con lunghezza minima = ',
      str(min_lenght), ': ', str(long_spans))
print('Media IoU:', str(mean_IoU))
print('Percentuale span perfetti: ', str(perfect_spans_perc))


print('----------------------------------------------------------')
print('Risultati labels GT e stralci uniti')


n_spans = 0
for e in test_df['Subword_bounds']:
    n_spans += len(e)
print('Numero stralci nel dataset:', str(n_spans))

n_spans = 0
for e in normalized_metrics:
    n_spans += len(e)
print('Numero stralci predetti:', str(n_spans))

mean = 0
long_spans = 0
min_lenght = 0
perfect_spans = 0
for text in normalized_metrics:
    for span in text:
        if span['Bounds'][1] - span['Bounds'][0] >= min_lenght:
            long_spans += 1
            mean += span['IoU']
            if span['IoU'] == 1:
                perfect_spans += 1
perfect_spans_perc = perfect_spans / long_spans
mean_IoU = mean / long_spans
print('Numero stralci con lunghezza minima = ',
      str(min_lenght), ': ', str(long_spans))
print('Media IoU:', str(mean_IoU))
print('Percentuale span perfetti: ', str(perfect_spans_perc))