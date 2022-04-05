import time
import datetime
from torch.nn import utils

import torchmetrics
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_constant_schedule_with_warmup

from utils.utils import format_time


class BertSegTrainer():

    def fit(self, model, train_dataset, val_dataset, batch_size, lr, n_epochs, loss_fn):

        output_dict = {}
        output_dict['train_metrics'] = []
        output_dict['train_loss'] = []
        output_dict['val_metrics'] = []
        output_dict['val_loss'] = []

        torch.cuda.empty_cache()
        # ----------TRAINING

        # Measure the total training time for the whole run.
        total_t0 = time.time()
        # Creation of Pytorch DataLoaders with shuffle=True for the traing phase
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True)

        # Adam algorithm optimized for tranfor architectures
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        #scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=300)

        # Scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler()

        # Setup for training with gpu
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        loss_fn.to(device)

        # For each epoch...
        for epoch_i in range(0, n_epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, n_epochs))
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
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(train_dataloader), elapsed))

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

                    #loss = outputs[0]
                    
                    logits = outputs[1]
                    print(logits.size())
                    print(logits.view(-1, model.num_labels).size())

                    print(b_labels.size())
                    print(b_labels.view(-1).size())
                    loss = loss_fn(logits.view(-1, model.num_labels), b_labels.view(-1))

                # Move logits and labels to CPU
                logits = logits.detach().cpu()

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
                # scheduler.step()

            # Compute the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            output_dict['train_loss'].append(avg_train_loss)

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
                    #loss = outputs[0]
                    
                    logits = outputs[1]

                    loss = loss_fn(logits.view(-1, model.num_labels), b_labels.view(-1))

                # Accumulate the validation loss.
                total_val_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu()

            print('VALIDATION: ')

            # Compute the average loss over all of the batches.
            avg_val_loss = total_val_loss / len(validation_dataloader)
            output_dict['val_loss'].append(avg_val_loss)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(
            format_time(time.time()-total_t0)))
        
        return output_dict

    def test(self, model, test_dataset, batch_size, loss_fn):
        # ========================================
        #               Test
        # ========================================
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        
        output_dict = {}

        # Setup for testing with gpu
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        loss_fn.to(device)

        print("")
        print("Running Test...")
        t0 = time.time()

        preds = []

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
                #loss = outputs[0]
                
                logits = outputs[1]
                loss = loss_fn(logits.view(-1, model.num_labels), b_labels.view(-1))

            # Accumulate the test loss.
            total_test_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu()  # shape (batch_size, seq_len, num_labels
            full_probs = logits.softmax(dim=-1)

            for i, sample_prob in enumerate(full_probs):
                active_prob = []
                for j, e in enumerate(b_special_tokens_mask[i]):
                    if(e == 0):
                        active_prob.append(sample_prob[j].tolist())
                preds.append(active_prob)

        avg_test_loss = total_test_loss / len(test_dataloader)
        test_time = format_time(time.time() - t0)

        output_dict['pred'] = preds
        output_dict['loss'] = avg_test_loss

        print("  Test Loss: {0:.2f}".format(avg_test_loss))
        print("  Test took: {:}".format(test_time))

        return output_dict