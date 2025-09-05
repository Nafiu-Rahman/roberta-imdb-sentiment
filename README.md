# IMDB Sentiment Classification (RoBERTa Fine-Tuned)

This project fine-tunes the **RoBERTa-base** transformer model on the IMDB movie reviews dataset to classify reviews as **positive** or **negative**.  
It includes training, evaluation, visualization, and deployment using a **Gradio web app** hosted on Hugging Face Spaces.

---

## Project Overview
- **Dataset**: [IMDB Reviews (50k reviews, balanced)](https://huggingface.co/datasets/imdb)  
- **Model**: [Model](https://huggingface.co/N4F1U/roberta-imdb-finetuned/tree/main)  
- **Goal**: Predict movie review sentiment with high accuracy.

## Demo
The model is deployed on Hugging Face Spaces: [roberta-imdb-sentiment](https://huggingface.co/spaces/N4F1U/roberta-imdb-sentiment)

## Demo Video
https://youtu.be/eYTaQEQLrP4

**Final Performance (on test set):**
- Accuracy: **94%**
- Precision: **0.94**
- Recall: **0.94**
- F1-score: **0.94**

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Nafiu-Rahman/roberta-imdb-sentiment.git
cd roberta-imdb-sentiment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install protobuf
```

### 3. Launch Gradio Demo
```bash
python app.py
```


