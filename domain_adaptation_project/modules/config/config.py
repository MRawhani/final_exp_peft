# config.py
import torch
class Config:
    DEBUG = False
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ### Text dataframe
    MIN_LENGTH = 400 # min number of characters per chunk
    MAX_LENGTH = 800 # max number of characters per chunk
    SOURCE_GENRE = 'telephone'
    TARGET_GENRE = 'government'
    SOURCE_DATA_LEN = 15000
    TARRGET_DATA_LEN = 25000


    ### model
    MODEL_NAME = 'bert-base-uncased' # roberta-base, distilroberta-base
    TOKENIZER_NAME = 'bert-base-uncased' # roberta-base, distilroberta-base
    
    ### split
    VALID_SIZE = 0.2
    SEED = 88    
    
    ### training
    LR = 2e-5
    EPOCHS = 2 if DEBUG else 3
    BATCH_SIZE = 32
    ACCUMULATION_STEPS = 4
    
    ### path to save tokenizer files
    RESULTS_SAVE_PATH = '../saved/results'
    TOKENIZER_SAVE_PATH = './hp-tokenizer-'
    
    ### path to save model files
    MODEL_SAVE_PATH = './hp-model-'
    ADAPTER_SAVE_PATH = '../saved/adapters'
    DATASETS_SAVE_PATH = '../../datasets'
    
    ### path to save txt files
    TXT_SAVE_PATH = './text-files/'
    # Add more configuration parameters as needed
