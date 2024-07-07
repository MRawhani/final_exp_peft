# data_utils.py
import config.config as cfg
from transformers import AutoTokenizer
# Import other necessary libraries
from torch.utils.data import Dataset, DataLoader

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from config.config import Config
from importlib import reload
import torch
import pytorch_lightning as pl
from datasets import load_from_disk
from typing import Optional, Dict, Any
from sklearn.model_selection import train_test_split
import pandas as pd


    
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


class DataModuleSourceTarget(pl.LightningDataModule):
    def __init__(self, hparams: Dict[str, Any]):
        super(DataModuleSourceTarget, self).__init__()
        #self.dataset_cache_dir = hparams["dataset_cache_dir"]
        self.source_target = hparams["source_target"]
        self.source_domain = hparams["source_domain"]
        self.target_domain = hparams["target_domain"]
        self.pretrained_model_name = hparams["pretrained_model_name"]
        self.padding = hparams["padding"]
        self.max_seq_length = hparams["max_seq_length"]
        self.batch_size = hparams["bsz"]

        # get the tokenizer using the pretrained_model_name that is required for transformers
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=True)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # No need to manually load files, use the datasets library
        pass

    def setup(self, stage: Optional[str] = None):
        reload(cfg)
        dataset = load_from_disk(f"../{cfg.Config.DATASETS_SAVE_PATH}/datasets")


        # Filter the dataset for different genres
        source_df = pd.DataFrame(dataset['train']).query(f"genre == '{self.source_domain}'")
        target_df = pd.DataFrame(dataset['train']).query(f"genre == '{self.target_domain}'")

        test_source_df = pd.DataFrame(dataset['validation_matched']).query(f"genre == '{self.source_domain}'")
        test_target_df = pd.DataFrame(dataset['validation_matched']).query(f"genre == '{self.target_domain}'")

        train_source_df, val_source_df = train_test_split(source_df, test_size=0.1, random_state=42,shuffle=True)
        train_target_df, val_target_df = train_test_split(target_df, test_size=0.1, random_state=42,shuffle=True)
        print(f"prinssst: {train_source_df.iloc[1]['genre']}")
        print(f"print: {train_target_df.iloc[1]['genre']}")
        print(f"print: {len(train_target_df)}")
        if stage == 'fit' or stage is None:
            self.train_dataset = SourceTargetDataset(train_source_df, train_target_df, self.tokenizer, self.padding, self.max_seq_length)
            self.val_dataset = SourceTargetDataset(val_source_df, val_target_df, self.tokenizer, self.padding, self.max_seq_length)
        if stage == 'test' or stage is None:
            self.test_dataset = SourceTargetDataset(test_source_df, test_target_df, self.tokenizer, self.padding, self.max_seq_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16)


class SourceTargetDataset(Dataset):
    def __init__(self, source_df, target_df, tokenizer, padding, max_seq_length):
        self.source_df = source_df
        self.target_df = target_df
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_seq_length = max_seq_length

    def __getitem__(self, index):
        premise = self.source_df.iloc[index]["premise"]
        hypothesis = self.source_df.iloc[index]["hypothesis"]
        label_source = self.source_df.iloc[index]["label"]

        encoded_input = self.tokenizer(
            str(premise),
            str(hypothesis),
            max_length=self.max_seq_length,
            truncation=True,
            padding=self.padding,
        )
        source_input_ids = encoded_input["input_ids"]
        source_attention_mask = encoded_input["attention_mask"]

        premise = self.target_df.iloc[index]["premise"]
        hypothesis = self.target_df.iloc[index]["hypothesis"]
        encoded_input = self.tokenizer(
            str(premise),
            str(hypothesis),
            max_length=self.max_seq_length,
            truncation=True,
            padding=self.padding,
        )
        target_input_ids = encoded_input["input_ids"]
        target_attention_mask = encoded_input["attention_mask"]

        if "label" in self.target_df.columns:
            label_target = self.target_df.iloc[index]["label"]
            data_input = {
                "source_input_ids": torch.tensor(source_input_ids),
                "source_attention_mask": torch.tensor(source_attention_mask),
                "target_input_ids": torch.tensor(target_input_ids),
                "target_attention_mask": torch.tensor(target_attention_mask),
                "label_source": torch.tensor(label_source, dtype=torch.long),
                "label_target": torch.tensor(label_target, dtype=torch.long),
            }
        else:
            data_input = {
                "source_input_ids": torch.tensor(source_input_ids),
                "source_attention_mask": torch.tensor(source_attention_mask),
                "target_input_ids": torch.tensor(target_input_ids),
                "target_attention_mask": torch.tensor(target_attention_mask),
                "label_source": torch.tensor(label_source, dtype=torch.long),
            }

        return data_input

    def __len__(self):
        return min(self.source_df.shape[0], self.target_df.shape[0])
