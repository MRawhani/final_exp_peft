from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification
from config.config import Config

def get_model_for_classification():
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME,num_labels=3)
    return model 
    