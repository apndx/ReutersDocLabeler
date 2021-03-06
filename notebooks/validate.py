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
from metrics import count_hits, evaluate
# from ReutersDocLabeler.notebooks.metrics import count_hits, evaluate

def test_time(start_time, end_time):
    # This is used to calculate the elapsed time
    # It was adopted from some of the homework assignments
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60.0)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60.0))
    return elapsed_mins, elapsed_secs

def test_model(device, model, model_name, criterion, num_labels, dataloader):
    # Parameters:
    # device: torch or cuda
    # model: the pretrained model to use
    # model_name: the name of the model to be used when it is saved for further use
    # criterion: the loss function to be used
    # num_labels: the number of the possible labels
    # dataloader: the dataloder that provides the input for the training

    print(f'Start validating model {model_name}')
    test_losses = []
    model.eval()
    steps = 0
    all_batch_losses = []
    scores = []
    # totals = pd.DataFrame(0, columns = list(range(0,126)), index = ['TP', 'TN', 'FP', 'FN'])
    totals = torch.zeros([4, 126], dtype = torch.int32)
    SCORE_INTERVAL = 10
    ALIVE_INTERVAL = 100
    
    # THE VALIDATION LOOP
    with torch.no_grad():
        test_start_time = time.time()
        total_loss = 0
        batch_losses = []
        for step, batch in enumerate(dataloader):
            batch_start_time = time.time()
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_token_types = batch # unpack from dataloader
            
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            loss = criterion(logits.view(-1, num_labels),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation

            # Update totals
            hits = count_hits(torch.sigmoid(logits), b_labels, 0.5)
            tp = hits['tp']
            tn = hits['tn']
            fp = hits['fp']
            fn = hits['fn']
            totals[0] = totals[0] + torch.sum(tp, 0).cpu().detach().numpy()
            totals[1] = totals[1] + torch.sum(tn, 0).cpu().detach().numpy()
            totals[2] = totals[2] + torch.sum(fp, 0).cpu().detach().numpy()
            totals[3] = totals[3] + torch.sum(fn, 0).cpu().detach().numpy()

            # Evaluate results
            if steps % SCORE_INTERVAL == 0 or steps == len(dataloader):
                score = evaluate(tp, tn, fp, fn)
                score['epoch'] = 1
                score['batch'] = step
                scores.append(score)

            total_loss += loss.item()
            steps += 1
            batch_end_time = time.time() 
            
            # Loss check
            loss_check = total_loss / (step+1)
            batch_losses.append(loss_check)
            batch_mins, batch_secs = test_time(batch_start_time, batch_end_time)
            if steps % ALIVE_INTERVAL == 0 or steps == len(dataloader):
                # Provide some feedback during the training
                print(f'Step {step} | Batch time: {batch_mins}m {batch_secs}s')
                print(f'\tLoss check: {loss_check:.3f}')
        
        test_loss = total_loss / len(dataloader)
        test_losses.append(test_loss)
        all_batch_losses.append(batch_losses)
            
        test_end_time = time.time()
            
        test_mins, test_secs = test_time(test_start_time, test_end_time)
        print(f'Testing Time: {test_mins}m {test_secs}s')
        print(f'\tTest Loss: {test_loss:.3f}')
        
    # Save the validation statistics for later analysis purposes
    run_id = model_name
    pdScores = pd.DataFrame(scores)
    pdScores.to_csv(f'notebooks/scores/scores_{run_id}.csv', index = False)
    batchLossDf = pd.DataFrame(all_batch_losses)
    batchLossDf.to_csv(f'notebooks/scores/batch_losses_{run_id}.csv', index = False)
    pdTotals = pd.DataFrame(totals.tolist(), index = ['TP', 'TN', 'FP', 'FN'])
    pdTotals.T.to_csv(f'notebooks/scores/totals_{run_id}.csv', index = False)


def main():
    # Arguments needed:
    # 1: Name of the dataloader file
    # 2: Name of the model file

    NUM_LABELS = 126 # amount of the different topics
    ADAM_DEFAULT_LR = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device in use:', device)
    test_data_loader_name = f'notebooks/data-loaders/{sys.argv[1]}'
    model_name = f'notebooks/models/{sys.argv[2]}'

    print(f'Load the validation data from {test_data_loader_name}')
    test_dataloader = torch.load(test_data_loader_name)

    print(f'Load the model from {model_name}')
    # model = BertForSequenceClassification.from_pretrained(model_name, num_labels = NUM_LABELS)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = NUM_LABELS)
    if device == 'cuda':
      model.load_state_dict(torch.load(model_name))
    else:
      model.load_state_dict(torch.load(model_name, map_location = torch.device('cpu')))
    model.to(device)
    criterion = BCEWithLogitsLoss()

    # Initiate validation of the model
    test_model(device, model, model_name, criterion, NUM_LABELS, test_dataloader)
    print('Finished')


if __name__ == "__main__":
    main()
