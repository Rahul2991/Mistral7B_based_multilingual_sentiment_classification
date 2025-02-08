from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from global_vars import DEVICE, MAX_LENGTH, NUM_CLASSES, label_mapping
from model import bnb_config
from torch.utils.data import DataLoader
from dataset import get_dataset
import torch, argparse, os
from tqdm import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataCollator:
    def __call__(self, features):
        # Convert to list of dictionaries
        batch = {key: [feature[key] for feature in features] for key in features[0].keys()}
        
        # Convert lists to tensors
        batch['input_ids'] = torch.tensor(batch['input_ids'], dtype=torch.long)
        batch['attention_mask'] = torch.tensor(batch['attention_mask'], dtype=torch.long)
        batch['label'] = torch.tensor(batch['label'], dtype=torch.long)
        
        return batch

data_collator = CustomDataCollator()

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader):
        inputs = {key: value.to(device) for key, value in batch.items() if key in tokenizer.model_input_names}
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    metrics = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation for Mistral 7B QLORA Fine-tuned on Multilingual Sentiment Classification Dataset clapAI/MultiLingualSentiment")
    parser.add_argument("-m", "--model_path", type=str, default="./results", help="Model Directory (default: ./results)")
    parser.add_argument("-subs_dt", "--subset_data", type=int, default=1, help="Subset Dataset for faster training (default: 1)")
    parser.add_argument("-ev_b", "--eval_batch_size", type=int, default=12, help="Batch size for validation (default: 12)")

    args = parser.parse_args()
    
    if not os.path.isdir(args.model_path): 
        print('Model path is invalid.')
        exit(1)
        
    print('Loading Model')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=NUM_CLASSES,
        quantization_config=bnb_config,
        ).to(DEVICE)
    
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.pad_token_id = model.config.eos_token_id
    model.enable_input_require_grads()
    
    print('Loading Model Successful')
    
    print('Fetching Dataset')
    _, _, test_dataset = get_dataset(subset=args.subset_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=data_collator)
    print('Dataset fetched successfully')

    print('Evaluating Model')
    metrics = evaluate_model(model, test_dataloader)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")