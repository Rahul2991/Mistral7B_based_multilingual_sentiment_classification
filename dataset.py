from global_vars import RANDOM_SEED
from datasets import load_dataset, DatasetDict, concatenate_datasets
from tokenizer import tokenize_function
import os, pickle

def sample_n_per_language(dataset, n):
    sampled_data = []
    for language in dataset.unique("language"):
        lang_subset = dataset.filter(lambda x: x["language"] == language)
        sampled_lang = lang_subset.shuffle(seed=RANDOM_SEED).select(range(min(n, len(lang_subset))))
        sampled_data.append(sampled_lang)
    return sampled_data
    
def get_dataset(subset=False):
    dataset = load_dataset("clapAI/MultiLingualSentiment")
    
    if subset:
        subset_path = 'subset_dataset.pkl'
        if os.path.exists(subset_path):
            with open(subset_path, 'rb') as f:
                dataset = pickle.load(f)
        else:       
            train_samples = sample_n_per_language(dataset["train"], 250)
            val_samples = sample_n_per_language(dataset["validation"], 150)
            test_samples = sample_n_per_language(dataset["test"], 100)
            
            dataset = DatasetDict({
                "train": concatenate_datasets(train_samples),
                "validation": concatenate_datasets(val_samples),
                "test": concatenate_datasets(test_samples),
            })
            with open(subset_path, 'wb') as f:
                pickle.dump(dataset, f)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets['validation']
    test_dataset = tokenized_datasets['test']
    
    return train_dataset, val_dataset, test_dataset