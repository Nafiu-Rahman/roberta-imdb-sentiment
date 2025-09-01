import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# Load model from local folder inside the Space
MODEL_PATH = "roberta-imdb-finetuned"

#tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained("N4F1U/roberta-imdb-finetuned")
model = AutoModelForSequenceClassification.from_pretrained("N4F1U/roberta-imdb-finetuned")

# Use GPU if available on the Space, otherwise CPU
device = 0 if torch.cuda.is_available() else -1

# Return scores for BOTH classes
pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=device,
    top_k=None,
    return_all_scores=True,
    function_to_apply="softmax",
)

# Map default HF labels to human-readable ones
label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}

def predict_sentiment(text: str):
    if not text or not text.strip():
        return {"negative": 0.0, "positive": 0.0}
    out = pipe(text)[0]  # list of dicts: {{'label': 'LABEL_0', 'score': 0.97}}, ...
    scores = { label_map.get(d["label"], d["label"]): float(d["score"]) for d in out }
    # gr.Label expects a dict of {{class_name: probability}}
    return scores

examples = [
    "A surprisingly moving film with excellent performances.",
    "This was painfully boring and I nearly fell asleep.",
    "Mixed feelings: great visuals, but the story dragged on.",
]

title = "IMDB Sentiment (RoBERTa-base fine-tuned)"
description = "Enter a movie review; the app returns probabilities for negative and positive."

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=6, placeholder="Type or paste an IMDB-style movie review here...", label="Review"),
    outputs=gr.Label(num_top_classes=2, label="Predicted sentiment"),
    examples=examples,
    title=title,
    description=description,
    flagging_mode="never",
)

# Queuing helps with concurrent requests on free CPU Spaces
demo.queue()

if __name__ == "__main__":

    demo.launch()
