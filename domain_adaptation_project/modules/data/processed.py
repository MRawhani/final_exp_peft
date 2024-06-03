# data_utils.py
import config.config as cfg
from transformers import AutoTokenizer
# Import other necessary libraries

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from config.config import Config
from importlib import reload

    
def tokenizes_and_load_datasets():
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
    reload(cfg)
    # Load datasets
    dataset = load_dataset('multi_nli')
    # Tokenize function
    def tokenize_function(examples):
     return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=128)

    # Tokenize datasets
    #filter by genre AND shuffle and select only portion of it based on predefined param
    filtered_source = dataset['train'].filter(lambda example: example['genre'] == Config.SOURCE_GENRE).train_test_split(test_size=0.1)
    
    shuffled_filtered_target = dataset['train'].filter(lambda example: example['genre'] == Config.TARGET_GENRE).shuffle(seed=42)
    # filtered_target = shuffled_filtered_target.select(range(Config.TARRGET_DATA_LEN)).train_test_split(test_size=0.1)
    filtered_target = shuffled_filtered_target.train_test_split(test_size=0.1)
    #unsupervised_target = shuffled_filtered_target.select(range(Config.TARRGET_DATA_LEN,shuffled_filtered_target.num_rows))
    
   
    filtered_test_target = dataset['validation_matched'].filter(lambda example: example['genre'] == Config.TARGET_GENRE)

 
    # Tokenize and filter datasets by genre
    tokenized_source = filtered_source['train'].map(tokenize_function, batched=True)
    tokenized_eval_source = filtered_source['test'].map(tokenize_function, batched=True)
    tokenized_test_target = filtered_test_target.map(tokenize_function, batched=True)
    tokenized_target = filtered_target['train'].map(tokenize_function, batched=True)
    tokenized_eval_target = filtered_target['test'].map(tokenize_function, batched=True)

    # Remove unused columns and set format to PyTorch tensors
    tokenized_source = tokenized_source.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre']).rename_column("label", "labels").with_format('torch')
    tokenized_target = tokenized_target.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre']).rename_column("label", "labels").with_format('torch')
    tokenized_eval_target = tokenized_eval_target.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre']).rename_column("label", "labels").with_format('torch')
    tokenized_eval_source = tokenized_eval_source.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre']).rename_column("label", "labels").with_format('torch')
    tokenized_test_target = tokenized_test_target.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre']).rename_column("label", "labels").with_format('torch')
   
    #load data
    source_loader = DataLoader(tokenized_source, batch_size=Config.BATCH_SIZE, shuffle=True,drop_last=True)
    source_loader_eval = DataLoader(tokenized_eval_source, batch_size=Config.BATCH_SIZE, shuffle=True,drop_last=True)
   
    target_loader = DataLoader(tokenized_target, batch_size=Config.BATCH_SIZE, shuffle=True,drop_last=True)
    target_loader_eval = DataLoader(tokenized_eval_target, batch_size=Config.BATCH_SIZE, shuffle=True,drop_last=True)
    target_loader_test = DataLoader(tokenized_test_target, batch_size=Config.BATCH_SIZE, shuffle=True,drop_last=True)

    # Return tokenized datasets and DataLoaders
    raw_data = {
        'source': filtered_source,
        'target': filtered_target,
        'test_target': filtered_test_target,
    }
    tokenized_data = {
        'source': tokenized_source,
        'eval_source': tokenized_eval_target,
        'target': tokenized_target,
        'eval_target': tokenized_eval_target,
        'test_target': tokenized_test_target,
    }

    loaded_data = {
        'source_loader': source_loader,
        'source_loader_eval': source_loader_eval,

        'target_loader': target_loader,
        'target_loader_eval': target_loader_eval,
        'test_target_loader': target_loader_test,
    }

    return tokenized_data, loaded_data,raw_data

def tokenize_dataset(data,tokenizer):
    def tokenize_function(examples):
        result = tokenizer(examples['premise'], examples['hypothesis']) # no trunccation or padding cuz it is mlm
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result


    # Use batched=True to activate fast multithreading!
    tokenized_datasets = data.map(
        tokenize_function, batched=True, 
    )
    tokenized_datasets = tokenized_datasets.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre','label'])

    return tokenized_datasets
