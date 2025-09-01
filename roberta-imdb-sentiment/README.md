# IMDB Sentiment (RoBERTa-base fine-tuned)

This Space hosts a Gradio demo for a RoBERTa-base model fine-tuned on the IMDB movie review dataset.
Enter a review and the app returns probabilities for **negative** and **positive** classes.

**How it works**
- Tokenize input with the same tokenizer used in training
- Run the sequence through the fine-tuned model
- Apply softmax and display per-class probabilities


Model files are bundled in this Space under `roberta-imdb-finetuned/`.
So in app.py you just need to change the lines:
- tokenizer = AutoTokenizer.from_pretrained("N4F1U/roberta-imdb-finetuned/tree/main/roberta-imdb-finetuned")
- model = AutoModelForSequenceClassification.from_pretrained("N4F1U/roberta-imdb-finetuned/tree/main/roberta-imdb-finetuned")



