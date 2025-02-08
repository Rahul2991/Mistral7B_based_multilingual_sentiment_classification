from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import bnb_config
import torch, argparse, os
from global_vars import DEVICE, MAX_LENGTH, NUM_CLASSES, label_mapping

def preprocess_text(text):
    return tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_LENGTH)

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def predict_sentiment(text):
    inputs = preprocess_text(text).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    predicted_class = torch.argmax(logits, dim=1).item()
        
    predicted_label = reverse_label_mapping[predicted_class]
    return predicted_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference for Mistral 7B QLORA Fine-tuned on Multilingual Sentiment Classification Dataset clapAI/MultiLingualSentiment")
    parser.add_argument("-m", "--model_path", type=str, default="./results", help="Model Directory (default: ./results)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.model_path): 
        print('Model path is invalid.')
        exit(1)
    
    text = None
    
    print('Loading Model')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=NUM_CLASSES,
        quantization_config=bnb_config,
        ).to(DEVICE)
    print('Loading Model Successful')

    print('Enter text to predict sentiment (type EXIT to quit):')
    while text != 'EXIT':
        text = input("Enter text to predict: ")
        if text == 'EXIT': break
        predicted_sentiment = predict_sentiment(text)
        print(f"Predicted sentiment: {predicted_sentiment}")