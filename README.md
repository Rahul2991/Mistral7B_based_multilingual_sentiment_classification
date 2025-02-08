# Mistral 7B QLORA Fine-tuning on Multilingual Sentiment Classification Dataset

This repository contains a PyTorch and Transformers implementation for QLORA fine-tuning the Mistral 7B Instruct model on the `clapAI/MultiLingualSentiment` dataset. The pipeline includes downloading the dataset, loading the pretrained model, training, saving the model and checkpoints, validating, evaluating, resuming training from a checkpoint, and performing inference.

## Features

- QLORA Fine-tune Mistral 7B Instruct on the `clapAI/MultiLingualSentiment` dataset.
- Save and load model checkpoints.
- Resume training from the last saved checkpoint.
- Perform inference on custom text inputs.
- Use PyTorch with CUDA support.
- Includes a `requirements.txt` for dependencies.

---

## Installation

### Prerequisites
Ensure you have Python 3.12.6 installed.

### Install PyTorch
To install PyTorch, visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and follow the instructions to select the appropriate version for your system and CUDA setup.

### Install Dependencies
Run the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Training the Model

### Steps
1. Clone this repository:
    ```bash
    git clone https://github.com/Rahul2991/Mistral7B_based_multilingual_sentiment_classification.git
    cd Mistral7B_based_multilingual_sentiment_classification
    ```

2. Start training:
    ```bash
    python train.py
    ```

3. Training output includes:
    - Checkpoints: Saved in the `checkpoints/` directory.
    - Final model: Saved in `results/` directory.

4. Resume training from a checkpoint:
    ```bash
    python train.py --resume_from_checkpoint checkpoints/<checkpoint_folder>
    ```
    
---

## Evaluation and Validation

After training, you can validate the model on the test dataset:

```bash
python evaluate_model.py
```

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

### Results
- Accuracy: 0.7135
- Precision: 0.7095
- Recall: 0.7135
- F1 Score: 0.7108

---

## Inference

To perform inference on custom text:

1. Run the inference script:
    ```bash
    python inference.py
    ```

2. The script will output the predicted sentiment label.

---

## Directory Structure

```plaintext
.
├── train.py             # Script for training the model
├── evaluate_model.py    # Script for evaluating the model
├── inference.py         # Script for running inference
├── requirements.txt     # Dependencies
├── results/             # Directory for saving the final model
├── checkpoints/         # Directory for saving training checkpoints
└── README.md            # Project documentation
```

---

## Example Usage

### Training
```bash
python train.py
```

### Resuming Training
```bash
python train.py --resume_training 1 --resume_training_checkpoint checkpoints/checkpoint-10000
```

### Evaluation
```bash
python evaluate_model.py
```

### Inference
```bash
python inference.py
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Hugging Face for the `transformers` and `datasets` libraries.
- `clapAI/MultiLingualSentiment` dataset for providing labeled emotion data.
- PyTorch for the deep learning framework.

---

## Citations
```bash
@dataset{clapAI2024multilingualsentiment,
  title        = {MultilingualSentiment: A Multilingual Sentiment Classification Dataset},
  author       = {clapAI},
  year         = {2024},
  url          = {https://huggingface.co/datasets/clapAI/MultiLingualSentiment},
  description  = {A multilingual dataset for sentiment analysis with labels: positive, neutral, negative, covering diverse languages and domains.},
}
```
