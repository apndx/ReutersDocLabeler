import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import sys


def main():

    # load csv
    df = pd.read_csv('notebooks/reuters-csv/inputs.csv', delimiter=';')
    print('Csv loaded')

    # change strings to lists
    df['target'] = df['target'].apply(eval)
    df['codes'] = df['codes'].apply(eval)

    documents = list(df.text.values)
    labels = list(df.target.values)
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)

    # encodings
    encodings = tokenizer.batch_encode_plus(
        documents, padding='max_length', truncation=False)  # tokenizer's encoding method
    input_ids = encodings['input_ids']  # tokenized and encoded sentences
    token_type_ids = encodings['token_type_ids']  # token type ids
    attention_masks = encodings['attention_mask']  # attention masks
    print('Encodings mapped')

    # rows that have unique targets
    label_counts = df.target.astype(str).value_counts()
    one_freq = label_counts[label_counts == 1].keys()
    one_freq_idxs = sorted(
        list(df[df.target.astype(str).isin(one_freq)].index), reverse=True)

    # Gathering single instance inputs
    one_freq_input_ids = [input_ids.pop(i) for i in one_freq_idxs]
    one_freq_token_types = [token_type_ids.pop(i) for i in one_freq_idxs]
    one_freq_attention_masks = [attention_masks.pop(i) for i in one_freq_idxs]
    one_freq_labels = [labels.pop(i) for i in one_freq_idxs]
    print('Rare targets separated')

    # original rows: 299 773. split train/dev/test 80/10/10, 29 977 for dev/test
    remaining_inputs, dev_inputs, remaining_labels, dev_labels, remaining_token_types, dev_token_types, remaining_masks, dev_masks = train_test_split(
        input_ids, labels, token_type_ids, attention_masks,
        random_state=42, test_size=0.101556, stratify=labels)
    print('Dev splits done')

    train_inputs, test_inputs, train_labels, test_labels, train_token_types, test_token_types, train_masks, test_masks = train_test_split(
        remaining_inputs, remaining_labels, remaining_token_types, remaining_masks,
        random_state=42, test_size=0.113035, stratify=remaining_labels)
    print('Test splits done')
    
    # add the unique rows to train
    train_inputs.extend(one_freq_input_ids)
    train_labels.extend(one_freq_labels)
    train_masks.extend(one_freq_attention_masks)
    train_token_types.extend(one_freq_token_types)
    print('Unique target rows added to train set')

    # change train sets to tensors
    t_train_inputs = torch.tensor(train_inputs)
    t_train_labels = torch.tensor(train_labels)
    t_train_masks = torch.tensor(train_masks)
    t_train_token_types = torch.tensor(train_token_types)

    # dev to tensors
    t_dev_inputs = torch.tensor(dev_inputs)
    t_dev_labels = torch.tensor(dev_labels)
    t_dev_masks = torch.tensor(dev_masks)
    t_dev_token_types = torch.tensor(dev_token_types)

    # test to tensors
    t_test_inputs = torch.tensor(test_inputs)
    t_test_labels = torch.tensor(test_labels)
    t_test_masks = torch.tensor(test_masks)
    t_test_token_types = torch.tensor(test_token_types)
    print('Data changed to tensors')
    print('train input:', t_train_inputs.shape)
    print('train labels:', t_train_labels.shape)
    print('dev input:', t_train_inputs.shape)
    print('test input:', t_train_inputs.shape)

    batch_size = 32

    # create train iterator with torch dataloader
    train_data = TensorDataset(t_train_inputs, t_train_masks, t_train_labels, t_train_token_types)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # create dev iterator with torch dataloader
    dev_data = TensorDataset(t_dev_inputs, t_dev_masks, t_dev_labels, t_dev_token_types)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

    # create test iterator with torch dataloader
    test_data = TensorDataset(t_test_inputs, t_test_masks, t_test_labels, t_test_token_types)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    print('Iterators created')

    # save loaders
    loader_name = sys.argv[1]
    torch.save(train_dataloader,f'notebooks/data-loaders/train_loader_{loader_name}')
    torch.save(dev_dataloader,f'notebooks/data-loaders/dev_loader_{loader_name}')
    torch.save(test_dataloader,f'notebooks/data-loaders/test_loader_{loader_name}')
    print('Dataloaders saved')

if __name__ == "__main__":
    main()
