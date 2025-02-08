from dotenv import load_dotenv
import numpy as np
import torch

load_dotenv()

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
NUM_CLASSES = 3
label_mapping = {"positive": 0, "neutral": 1, "negative": 2}
MAX_LENGTH = 512 
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.cuda.manual_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

