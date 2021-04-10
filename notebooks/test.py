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
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60.0)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60.0))
    return elapsed_mins, elapsed_secs

def test_model(device, model, model_name, num_labels, dataloader, itemids):
    print(f'Start testing model {model_name}')
    model.eval()
    steps = 0
    ALIVE_INTERVAL = 100
    results = []
    
    with torch.no_grad():
        test_start_time = time.time()
        for step, batch in enumerate(dataloader):
            batch_start_time = time.time()
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_token_types = batch # unpack from dataloader
            
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]

            prediction = logits > 0.5
            results.extend(prediction)

            steps += 1
            batch_end_time = time.time() 
            
            # Alive check
            batch_mins, batch_secs = test_time(batch_start_time, batch_end_time)
            if steps % ALIVE_INTERVAL == 0 or steps == len(dataloader):
                print(f'Step {step} | Batch time: {batch_mins}m {batch_secs}s')
        
        test_end_time = time.time()
            
        test_mins, test_secs = test_time(test_start_time, test_end_time)
        print(f'Testing Time: {test_mins}m {test_secs}s')
        
    run_id = sys.argv[2]
    dfResults = pd.DataFrame(results)
    dfResults['id'] = itemids
    dfResults.to_csv(f'notebooks/test_results_{run_id}.csv', index = False)


def main():

    NUM_LABELS = 126 # amount of the different topics
    ADAM_DEFAULT_LR = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device in use:', device)
    test_data_loader_name = f'notebooks/data-loaders/{sys.argv[1]}'
    model_name = f'notebooks/models/{sys.argv[2]}'

    print(f'Load the test data from {test_data_loader_name}')
    test_dataloader = torch.load(test_data_loader_name)

    print(f'Load the model from {model_name}')

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = NUM_LABELS)
    if device == 'cuda':
      model.load_state_dict(torch.load(model_name))
    else:
      model.load_state_dict(torch.load(model_name, map_location = torch.device('cpu')))
    model.to(device)

    # load csv
    df = pd.read_csv(f'notebooks/reuters-csv/test.csv', delimiter=';')
    itemids = df['id']

    test_model(device, model, model_name, NUM_LABELS, test_dataloader, itemids)
    print('Finished')


if __name__ == "__main__":
    main()
