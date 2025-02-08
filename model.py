from transformers import AutoModelForSequenceClassification
from tokenizer import get_tokenizer
from global_vars import model_name, NUM_CLASSES
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
import torch, os

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        token=os.getenv('HF_TOKEN'),
        quantization_config=bnb_config,
    )
    
    model.resize_token_embeddings(len(get_tokenizer()))
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.pad_token_id = model.config.eos_token_id
    model.enable_input_require_grads()
    
    modules = find_all_linear_names(model)

    lora_config = LoraConfig(
        r=16,  # Rank of the low-rank update matrices
        lora_alpha=32,  # Scaling factor for LoRA updates
        target_modules=modules,  # Layers to apply LoRA
        lora_dropout=0.1,  
        bias="none",  
        task_type=TaskType.SEQ_CLS
    )
    
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    return model