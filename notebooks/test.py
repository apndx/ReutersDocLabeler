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

def test_model(device, model, model_name, num_labels, dataloader, itemids, topic_codes):
    # Parameters:
    # device: torch or cuda
    # model: the pretrained model to use
    # model_name: the name of the model to be used when it is saved for further use
    # num_labels: the number of the possible labels
    # dataloader: the dataloder that provides the input for the training
    # itemids: the ids of the news items for reporting the prediction results
    # topic_codes: the topics_codes for reporting the prediction results

    print(f'Start testing model {model_name}')
    model.eval()
    steps = 0
    ALIVE_INTERVAL = 100
    results = []
    
    # THE TESTING LOOP
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
            result = np.array(prediction.tolist())
            results.extend(result)

            steps += 1
            batch_end_time = time.time() 
            
            # Alive check
            batch_mins, batch_secs = test_time(batch_start_time, batch_end_time)
            if steps % ALIVE_INTERVAL == 0 or steps == len(dataloader):
                print(f'Step {step} | Batch time: {batch_mins}m {batch_secs}s')
        
        test_end_time = time.time()
            
        test_mins, test_secs = test_time(test_start_time, test_end_time)
        print(f'Testing Time: {test_mins}m {test_secs}s')
    
    # Make the result file to be returned
    # Columns:
    # id = the newsitem id
    # columns 1 to 126 named after the topic codes
    # values: 0s and 1s with 1 marking a topic code is predicted to be present
    run_id = sys.argv[2]
    dfResults = pd.DataFrame(results)
    dfResults = dfResults + 0
    dfResults.columns = topic_codes
    dfResults.insert(loc = 0, column = 'id', value = itemids)
    dfResults.to_csv(f'notebooks/scores/test_results_{run_id}.csv', sep = ' ', index = False)


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

    print(f'Load the test data from {test_data_loader_name}')
    test_dataloader = torch.load(test_data_loader_name)

    print(f'Load the model from {model_name}')

    # Not sure this is the most straight forward way, but it works
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = NUM_LABELS)
    if device == 'cuda':
      model.load_state_dict(torch.load(model_name))
    else:
      model.load_state_dict(torch.load(model_name, map_location = torch.device('cpu')))
    model.to(device)

    # load csv
    df = pd.read_csv(f'notebooks/reuters-csv/test.csv', delimiter=';')
    itemids = df['id']
    topics = pd.read_csv('notebooks/reuters-csv/topic_codes.txt', delimiter='\t')
    topic_codes = topics['CODE'].tolist()

    # Initiate testing
    test_model(device, model, model_name, NUM_LABELS, test_dataloader, itemids, topic_codes)
    print('Finished')


if __name__ == "__main__":
    main()
