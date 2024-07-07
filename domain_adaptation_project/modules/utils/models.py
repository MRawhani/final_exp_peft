# Step 4: Define the DomainTaskAdapter class
import torch.nn as nn
import torch.optim as optim
from transformers import AutoConfig
from adapters import AutoAdapterModel, AdapterConfig
from adapters.composition import Stack
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import torchmetrics
import torch
import pytorch_lightning as pl
import pandas as pd
from transformers import AutoTokenizer



class DomainTaskAdapter(pl.LightningModule):
    def __init__(self, hparams):
        super(DomainTaskAdapter, self).__init__()
        self.save_hyperparameters(hparams)
        self.config = AutoConfig.from_pretrained(self.hparams["pretrained_model_name"])
        self.config.output_hidden_states = True
        self.model = AutoAdapterModel.from_pretrained(self.hparams["pretrained_model_name"], config=self.config)
        
        self.reduction_factor = self.hparams.get("reduction_factor", 16)
        if self.reduction_factor == "None":
            self.reduction_factor = 16
        self.leave_out = self.hparams.get("leave_out", [])
        #if self.leave_out != "None":
         #   self.leave_out = [int(i) for i in self.leave_out.split(",")]

        adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=self.reduction_factor, leave_out=self.leave_out)
        
        #self.model.add_adapter("task_adapter_{}".format(self.hparams["source_target"]), config=adapter_config)
        self.dapter_name = self.hparams["adapter_name"]
        
        if self.hparams["mode"] == "domain":
            self.model.load_adapter(f"../saved/adapters/{adapter_name}", with_head=False)
            self.model.add_classification_head(adapter_name, num_labels=self.hparams["num_classes"])
            self.model.active_adapters = adapter_name
            self.model.train_adapter(adapter_name)
        else:
            self.model.train_adapter("task_adapter_{}".format(self.hparams["source_target"]))
        self.validation_outputs = []
        self.test_outputs = []
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass',                                           
                                     num_classes=self.hparams["num_classes"])
        self.f1 = torchmetrics.F1Score(task='multiclass',num_classes=self.hparams["num_classes"], average="macro")
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        labels = batch["label_source"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits), dim=1))
        f1 = self.f1(labels, torch.argmax(self.softmax(logits), dim=1))

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        self.log("train_f1", f1)
        return loss
    def validation_step(self, batch, batch_idx):
        """validation step of DomainTaskAdapter"""
        # get the input ids and attention mask for source data
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch["label_source"]
        source_loss = self.criterion(logits, labels)
        source_accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits), dim=1))
        source_f1 = self.f1(labels, torch.argmax(self.softmax(logits), dim=1))

        # get the input ids and attention mask for target data
        input_ids, attention_mask = batch["target_input_ids"], batch["target_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch["label_target"]
        target_loss = self.criterion(logits, labels)
        target_accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits), dim=1))
        target_f1 = self.f1(labels, torch.argmax(self.softmax(logits), dim=1))

     
        # this will log the mean div value across epoch
        self.log(name="source_val/loss", value=source_loss, prog_bar=True, logger=True)
        self.log(name="source_val/accuracy", value=source_accuracy, prog_bar=True, logger=True)
        self.log(name="target_val/loss", value=target_loss, prog_bar=True, logger=True)
        self.log(name="target_val/accuracy", value=target_accuracy, prog_bar=True, logger=True)
        self.log(name="target_val/f1", value=target_f1, prog_bar=True, logger=True)
        self.log(name="source_val/f1", value=source_f1, prog_bar=True, logger=True)
        self.validation_outputs.append({
            "source_val/loss": source_loss,
            "source_val/accuracy": source_accuracy,
            "source_val/f1": source_f1,
            "target_val/loss": target_loss,
            "target_val/accuracy": target_accuracy,
            "target_val/f1": target_f1,
            })
        return {
            "source_val/loss": source_loss,
            "source_val/accuracy": source_accuracy,
            "source_val/f1": source_f1,
            "target_val/loss": target_loss,
            "target_val/accuracy": target_accuracy,
            "target_val/f1": target_f1,
        }
    def on_validation_epoch_start(self):
        self.validation_outputs = []
    
    def on_validation_epoch_end(self):
        outputs= self.validation_outputs
        mean_source_loss = torch.stack([x["source_val/loss"] for x in outputs]).mean()
        mean_source_accuracy = torch.stack([x["source_val/accuracy"] for x in outputs]).mean()
        mean_source_f1 = torch.stack([x["source_val/f1"] for x in outputs]).mean()

        mean_target_loss = torch.stack([x["target_val/loss"] for x in outputs]).mean()
        mean_target_accuracy = torch.stack([x["target_val/accuracy"] for x in outputs]).mean()
        mean_target_f1 = torch.stack([x["target_val/f1"] for x in outputs]).mean()
        print(f"target_val/loss: {mean_target_loss}")
        print(f"target_val/accuracy: {mean_target_accuracy}")
        print(f"target_val/f1: {mean_target_accuracy}")
        print(f"source_val/loss: {mean_target_loss}")
        print(f"source_val/accuracy: {mean_target_accuracy}")
        print(f"source_val/f1: {mean_target_accuracy}")
        # this will log the mean div value across epoch
        self.log(name="source_val/loss", value=mean_source_loss, prog_bar=True, logger=True)
        self.log(name="source_val/accuracy", value=mean_source_accuracy, prog_bar=True, logger=True)
        self.log(name="target_val/loss", value=mean_target_loss, prog_bar=True, logger=True)
        self.log(name="target_val/accuracy", value=mean_target_accuracy, prog_bar=True, logger=True)
        self.log(name="target_val/f1", value=mean_target_f1, prog_bar=True, logger=True)
        self.log(name="source_val/f1", value=mean_source_f1, prog_bar=True, logger=True)
                # Log `val_loss` as `mean_source_loss`
        self.log("val_loss", mean_source_loss)

    def test_step(self, batch, batch_idx):
        """validation step of DomainTaskAdapter"""
        # get the input ids and attention mask for source data
        input_ids, attention_mask = batch["source_input_ids"], batch["source_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch["label_source"]
        source_loss = self.criterion(logits, labels)
        source_accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits), dim=1))
        source_f1 = self.f1(labels, torch.argmax(self.softmax(logits), dim=1))

        # get the input ids and attention mask for target data
        input_ids, attention_mask = batch["target_input_ids"], batch["target_attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        labels = batch["label_target"]
        target_loss = self.criterion(logits, labels)
        target_accuracy = self.accuracy(labels, torch.argmax(self.softmax(logits), dim=1))
        target_f1 = self.f1(labels, torch.argmax(self.softmax(logits), dim=1))

        # this will log the mean div value across epoch
        self.log(name="source_test/loss", value=source_loss)
        self.log(name="source_test/accuracy", value=source_accuracy)
        self.log(name="target_test/loss", value=target_loss)
        self.log(name="target_test/accuracy", value=target_accuracy)
        self.log(name="target_test/f1", value=target_f1)
        self.log(name="source_test/f1", value=source_f1)
        self.test_outputs.append({
            "source_test/loss": source_loss,
            "source_test/accuracy": source_accuracy,
            "source_test/f1": source_f1,
            "target_test/loss": target_loss,
            "target_test/accuracy": target_accuracy,
            "target_test/f1": target_f1,
         })
        # need not to log here (or we can do it but let's log at the end of each epoch)
        return {
            "source_test/loss": source_loss,
            "source_test/accuracy": source_accuracy,
            "source_test/f1": source_f1,
            "target_test/loss": target_loss,
            "target_test/accuracy": target_accuracy,
            "target_test/f1": target_f1,
        }
    def on_test_epoch_start(self):
        self.test_outputs = []
    def on_test_epoch_end(self):
        outputs=  self.test_outputs
        mean_source_loss = torch.stack([x["source_test/loss"] for x in outputs]).mean()
        mean_source_accuracy = torch.stack([x["source_test/accuracy"] for x in outputs]).mean()
        mean_source_f1 = torch.stack([x["source_test/f1"] for x in outputs]).mean()

        mean_target_loss = torch.stack([x["target_test/loss"] for x in outputs]).mean()
        mean_target_accuracy = torch.stack([x["target_test/accuracy"] for x in outputs]).mean()
        mean_target_f1 = torch.stack([x["target_test/f1"] for x in outputs]).mean()

        # this will log the mean div value across epoch
        self.log(name="source_test/loss", value=mean_source_loss)
        self.log(name="source_test/accuracy", value=mean_source_accuracy)
        self.log(name="target_test/loss", value=mean_target_loss)
        self.log(name="target_test/accuracy", value=mean_target_accuracy)
        self.log(name="target_test/f1", value=mean_target_f1)
        self.log(name="source_test/f1", value=mean_source_f1)
    def save_adapter(self, location, adapter_name):
        self.model.save_adapter(location, adapter_name)
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams["learning_rate"])
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001, cooldown=0, min_lr=1e-8),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]
