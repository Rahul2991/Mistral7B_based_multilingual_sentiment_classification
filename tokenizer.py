from transformers import AutoTokenizer
from global_vars import model_name, label_mapping, MAX_LENGTH

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token
    
get_tokenizer = lambda: tokenizer

def tokenize_function(inp):
    inp["label"] = [label_mapping[label] for label in inp["label"]]
    return tokenizer(inp['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)