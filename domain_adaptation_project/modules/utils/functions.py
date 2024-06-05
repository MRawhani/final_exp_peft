import torch
from config.config import Config
from sklearn.metrics import accuracy_score, f1_score,precision_recall_fscore_support
import numpy as np
from transformers import TrainingArguments, EvalPrediction,default_data_collator, Trainer
from adapters import AdapterTrainer
from transformers import EvalPrediction,EarlyStoppingCallback,get_cosine_schedule_with_warmup
import collections
import re 

import os
def evaluate_model(model, dataloader):
    model.to(Config.DEVICE)
    model.eval()
    predictions, true_labels = [], []
    for valid_step, batch in enumerate(dataloader):

        with torch.no_grad():
            batch = {k: v.to(Config.DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, f1


def print_trainable_parameters(model):

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def train_model(model,prepended_path,train_data, eval_data=None):
    training_args = TrainingArguments(
        
        output_dir=f"{Config.RESULTS_SAVE_PATH}/{prepended_path}/results",                 # Where to store the output (checkpoints and predictions)
        num_train_epochs=4,                     # Total number of training epochs
        per_device_train_batch_size=32,         # Batch size for training
        per_device_eval_batch_size=32,          # Batch size for evaluation
        warmup_steps=500,                       # Number of warmup steps for learning rate scheduler
        learning_rate=1e-4,
        weight_decay=0.01,                      # Strength of weight decay
        logging_dir=f"{Config.RESULTS_SAVE_PATH}/{prepended_path}/logs",                   # Directory for storing logs
        logging_steps=500,                       # Log every X updates steps
        evaluation_strategy="steps" if eval_data is not None else "no",            # Evaluate model every X steps
        eval_steps=500,                         # Number of steps to perform evaluation
        save_steps=500,                         # Save checkpoint every X steps
        save_total_limit=2,                     # Limit the total amount of checkpoints
        load_best_model_at_end=True if eval_data is not None else False,            # Load the best model when finished training
        metric_for_best_model="accuracy",       # Use accuracy to find the best model
        greater_is_better=True,                 # Higher accuracy is better
        report_to="none"                        # Do not report to any online service
    ) 
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    trainer = AdapterTrainer(
        model=model,                           # The instantiated 🤗 Transformers model to be trained
        args=training_args,                    # Training arguments, defined above
        train_dataset=train_data,           # Training dataset
        eval_dataset=eval_data if eval_data is not None else None,
        compute_metrics=compute_metrics if eval_data is not None else None    )
    trainer.train()
    return trainer

def train_mlm_model(model,prepended_path,collator, tokenizer,train_data, eval_data=None):
    batch_size = 32

    logging_steps = len(train_data) // batch_size

    training_args = TrainingArguments(
        
        output_dir=f"{Config.RESULTS_SAVE_PATH}/{prepended_path}/results",                 # Where to store the output (checkpoints and predictions)
        num_train_epochs=20,                     # Total number of training epochs
        per_device_train_batch_size=batch_size,         # Batch size for training
        per_device_eval_batch_size=batch_size,          # Batch size for evaluation
        warmup_steps=500,                       # Number of warmup steps for learning rate scheduler
        learning_rate=1e-4,
        weight_decay=0.01,                      # Strength of weight decay
        logging_dir=f"{Config.RESULTS_SAVE_PATH}/{prepended_path}/logs",                   # Directory for storing logs
        logging_steps=logging_steps,                       # Log every X updates steps
        evaluation_strategy="epoch" if eval_data is not None else "no",            # Evaluate model every X steps
        save_strategy="epoch" if eval_data is not None else "no",            # Evaluate model every X steps
        eval_steps=logging_steps,                         # Number of steps to perform evaluation
        save_steps=logging_steps,                         # Save checkpoint every X steps
        save_total_limit=2,                     # Limit the total amount of checkpoints
        fp16=True,
        metric_for_best_model="eval_loss",  # Use perplexity to determine the best model
        load_best_model_at_end=True if eval_data is not None else False,            # Load the best model when finished training
        #remove_unused_columns=False,#for collator word id

        greater_is_better=False,                 # lower  perplexity is better
        report_to="none"                        # Do not report to any online service
    )
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)
        non_masked_tokens = shift_labels.ne(-100).sum().item()
        perplexity = torch.exp(loss / non_masked_tokens)
        return {"perplexity": perplexity.item()}
   
    

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = AdapterTrainer(
        model=model,                           # The instantiated 🤗 Transformers model to be trained
        args=training_args,                    # Training arguments, defined above
        train_dataset=train_data,           # Training dataset
        eval_dataset=eval_data if eval_data is not None else None,
        data_collator=collator,
        tokenizer=tokenizer,
            callbacks=callbacks

        # compute_metrics=compute_metrics
            )
    # Create the optimizer and scheduler
    trainer.create_optimizer_and_scheduler(num_training_steps=len(train_data) * training_args.num_train_epochs // batch_size)

    # Get the optimizer and replace its scheduler with the custom one
    optimizer = trainer.optimizer
    total_steps = len(train_data) * training_args.num_train_epochs // batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=total_steps)

    trainer.lr_scheduler = scheduler


    return trainer

def train_mlm_model_without_adapter(model,prepended_path,collator, tokenizer,train_data, eval_data=None):
    batch_size = 32

    logging_steps = len(train_data) // batch_size

    training_args = TrainingArguments(
        
        output_dir=f"{Config.RESULTS_SAVE_PATH}/{prepended_path}/results",                 # Where to store the output (checkpoints and predictions)
        num_train_epochs=10,                     # Total number of training epochs
        per_device_train_batch_size=batch_size,         # Batch size for training
        per_device_eval_batch_size=batch_size,          # Batch size for evaluation
        warmup_steps=500,                       # Number of warmup steps for learning rate scheduler
        learning_rate=1e-4,
        weight_decay=0.01,                      # Strength of weight decay
        logging_dir=f"{Config.RESULTS_SAVE_PATH}/{prepended_path}/logs",                   # Directory for storing logs
        logging_steps=logging_steps,                       # Log every X updates steps
        evaluation_strategy="epoch" if eval_data is not None else "no",            # Evaluate model every X steps
        save_strategy="epoch" if eval_data is not None else "no",            # Evaluate model every X steps
        eval_steps=logging_steps,                         # Number of steps to perform evaluation
        save_steps=logging_steps,                         # Save checkpoint every X steps
        save_total_limit=2,                     # Limit the total amount of checkpoints
        fp16=True,
        metric_for_best_model="eval_loss",  # Use perplexity to determine the best model
        load_best_model_at_end=True if eval_data is not None else False,            # Load the best model when finished training
        #remove_unused_columns=False,#for collator word id

        greater_is_better=False,                 # lower  perplexity is better
        report_to="none"                        # Do not report to any online service
    )
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)
        non_masked_tokens = shift_labels.ne(-100).sum().item()
        perplexity = torch.exp(loss / non_masked_tokens)
        return {"perplexity": perplexity.item()}
    trainer = Trainer(
        model=model,                           # The instantiated 🤗 Transformers model to be trained
        args=training_args,                    # Training arguments, defined above
        train_dataset=train_data,           # Training dataset
        eval_dataset=eval_data if eval_data is not None else None,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
            )
    
    return trainer

def group_texts(examples,chunk_size):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result



def whole_word_masking_data_collator(features,tokenizer):   
    wwm_probability = 0.15

    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)



def count_words(dataset):
    total_premise_words = 0
    total_hypothesis_words = 0

    # Iterate through each record in the dataset
    for entry in dataset:
        # Split the 'premise' and 'hypothesis' fields on spaces to count words
        premise_words = entry['premise'].split()
        hypothesis_words = entry['hypothesis'].split()

        # Sum up the word counts
        total_premise_words += len(premise_words)
        total_hypothesis_words += len(hypothesis_words)

    return total_premise_words + total_hypothesis_words
def count_sentences_basic(dataset):
    total_premise_sentences = 0
    total_hypothesis_sentences = 0

    # Define a function to count sentences based on specific punctuation marks
    def sentence_count(text):
        # Split the text by sentence-ending punctuation followed by space or end of string
        sentences = [s.strip() for s in re.split(r'[.?!]+\s+|$', text) if s.strip()]
        return len(sentences)

    # Iterate through each record in the dataset
    for entry in dataset:
        # Count sentences in the 'premise' and 'hypothesis' fields
        total_premise_sentences += sentence_count(entry['premise'])
        total_hypothesis_sentences += sentence_count(entry['hypothesis'])

    return total_premise_sentences + total_hypothesis_sentences



def train_mlm_model_with_hyperparameters(model,prepended_path,collator, tokenizer,train_data, eval_data,
    learning_rate=1e-3, weight_decay=0.01, num_train_epochs=10, warmup_steps=500
):
    batch_size = int(32)
    
    logging_steps = len(train_data) // batch_size
    num_train_epochs = int(num_train_epochs)
    warmup_steps = int(warmup_steps)
    training_args = TrainingArguments(
        output_dir=f"{Config.RESULTS_SAVE_PATH}/{prepended_path}/results",                 # Where to store the output (checkpoints and predictions)
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_dir=f"{Config.RESULTS_SAVE_PATH}/{prepended_path}/logs",                   # Directory for storing logs
        logging_steps=logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        gradient_accumulation_steps=4,  # Simulates a larger batch size
        
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        report_to="none"
    )

    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)
        non_masked_tokens = shift_labels.ne(-100).sum().item()
        perplexity = torch.exp(loss / non_masked_tokens)
        return {"perplexity": perplexity.item()}

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        compute_metrics=compute_metrics
    )

    trainer.create_optimizer_and_scheduler(num_training_steps=len(train_data) * num_train_epochs // batch_size)
    
    optimizer = trainer.optimizer
    total_steps = len(train_data) * num_train_epochs // batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    trainer.lr_scheduler = scheduler

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results