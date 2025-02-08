from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from dataset import get_dataset
from model import get_model
from tokenizer import get_tokenizer
from evaluate import compute_metrics
import argparse, os

def main():
    parser = argparse.ArgumentParser(description="Mistral 7B QLORA Fine-tuning on Multilingual Sentiment Classification Dataset clapAI/MultiLingualSentiment")
    
    parser.add_argument("-tb", "--train_batch_size", type=int, default=12, help="Batch size for training (default: 12)")
    parser.add_argument("-vb", "--val_batch_size", type=int, default=12, help="Batch size for validation (default: 12)")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of Epochs (default: 1)")
    parser.add_argument("-s", "--save_steps", type=int, default=50, help="Save checkpoint after steps (default: 50)")
    parser.add_argument("-eval_stps", "--eval_steps", type=int, default=50, help="Eval after steps (default: 50)")
    parser.add_argument("-c", "--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints (default: ./checkpoints)")
    parser.add_argument("-res", "--results_dir", type=str, default="./results", help="Directory to save final results (default: ./results)")
    parser.add_argument("-eval_st", "--eval_strategy", type=str, choices=['epoch', 'steps', 'no'], default='steps', help="Evaluation strategy: 'epoch', 'steps', or 'no' (default: 'steps')")
    parser.add_argument("-sav_st", "--save_strategy", type=str, choices=['epoch', 'steps', 'no'], default='steps', help="Save checkpoint strategy: 'epoch', 'steps', or 'no' (default: 'steps')")
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5, help="Learning rate for training (default: 5e-5)")
    parser.add_argument("-subs_dt", "--subset_data", type=int, default=1, help="Subset Dataset for faster training (default: 1)")
    parser.add_argument("-res_tr", "--resume_training", type=int, default=0, help="Resume training from previous checkpoint (default: 0)")
    parser.add_argument("-res_chkpt", "--resume_training_checkpoint", type=str, default="None", help="Resume training from previous checkpoint location (default: None) Eg.'checkpoints/checkpoint-10000'")

    args = parser.parse_args()
    
    if args.resume_training and args.resume_training_checkpoint == "None": 
        print('Resume training checkpoint is required.')
        exit(1)

    optim = "paged_adamw_32bit"
    lr_scheduler_type = "cosine"
    
    training_args = TrainingArguments(
        gradient_accumulation_steps=1,
        bf16=True,
        output_dir=args.checkpoint_dir,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        save_steps=args.save_steps,  
        save_on_each_node=True,
        gradient_checkpointing=True,
        weight_decay=0.01,
        save_strategy=args.save_strategy,
        logging_dir="./logs",
        logging_steps=10,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
    )
    
    print('Training Args Set Successfully')
    
    print('Fetching Dataset')
    train_dataset, val_dataset, test_dataset = get_dataset(subset=args.subset_data)
    print('Dataset fetched successfully')
    
    print('Initializing Trainer')
    trainer = Trainer(
        model=get_model(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    print('Trainer Initialized Successfully')
    
    if args.resume_training:
        if not os.path.isdir(args.resume_training_checkpoint): 
            print('Resume training checkpoint is invalid.')
            exit(1)
        print('Resuming Training from Previous Checkpoint')
        try:
            trainer.train(resume_from_checkpoint=args.resume_training_checkpoint)
        except Exception as e:
            print(f"Error occurred while resuming training: {e}")
            exit(1)
    else:
        print('Starting Training')
        trainer.train()
    print('Training Completed Successfully')
    
    trainer.save_state()
    print('Saving Trainer State')
    
    trainer.save_model(output_dir=args.results_dir)
    print('Saving Trained Model')
    
    get_tokenizer().save_pretrained(save_directory=args.results_dir)  ## Tokenizer

    print('Evaluating Trained Model')
    test_results = trainer.evaluate(test_dataset)
    print(f"Evalutaion Results: {test_results}")
    
    trainer.save_metrics(split='test', metrics=test_results)
    print('Saving Metrics')
    
    print('Done')
    
if __name__ == '__main__':
    main()