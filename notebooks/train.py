import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
import sys

def define_model(device, num_labels, lr):
    print('Start defining the model')
    model_1 = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    model_1.to(device)
    optimizer_1 = AdamW(model_1.parameters(), lr=lr)
    criterion_1 = BCEWithLogitsLoss()
    return model_1, optimizer_1, criterion_1

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_model(device,model, model_name, optimizer, criterion, n_epochs, num_labels, dataloader):
    print('Start training')
    train_losses = []
    model.train()
    steps = 0
    examples = 0
    all_batch_losses = []
    
    for epoch in range(n_epochs):
        start_time = time.time()
        epoch_loss = 0
        batch_losses = []
        for step, batch in enumerate(dataloader):
            if step == 3:
                break
            batch_start_time = time.time()
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_token_types = batch # unpack from dataloader
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            loss = criterion(logits.view(-1, num_labels),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation

            # Backward pass
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            examples += b_input_ids.size(0)
            steps += 1
            batch_end_time = time.time() 
            
            # Loss check
            loss_check = epoch_loss/(step+1)
            batch_losses.append(loss_check)
            batch_mins, batch_secs = epoch_time(batch_start_time, batch_end_time)
            print(f'Epoch: {epoch+1:02} | Step {step} | Batch time: {batch_mins}m {batch_secs}s')
            print(f'\tLoss check: {loss_check:.3f}')
        
        torch.save(model.state_dict(), model_name)    
        train_loss = epoch_loss / len(dataloader)
        train_losses.append(train_loss)
        all_batch_losses.append(batch_losses)
            
        end_time = time.time()
            
        epoch_mins, epoch_secs = epoch_time(batch_start_time, batch_end_time)
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        
            
    return train_losses, all_batch_losses



def main():

    NUM_LABELS = 126 # amount of the different topics
    ADAM_DEFAULT_LR = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data_loader_name = f'notebooks/data-loaders/{sys.argv[1]}'
    model_name = f'notebooks/models/{sys.argv[1]}.pt'

    train_dataloader = torch.load(train_data_loader_name)
    model, optimizer, criterion = define_model(device, NUM_LABELS, ADAM_DEFAULT_LR)
  
    n_epochs = 1

    train_losses, all_batch_losses = train_model(device, model, model_name, optimizer, criterion, n_epochs, NUM_LABELS, train_dataloader)
    print('Finished')
    print('Train losses:', train_losses)
    print('Batch losses:', all_batch_losses)


if __name__ == "__main__":
    main()
