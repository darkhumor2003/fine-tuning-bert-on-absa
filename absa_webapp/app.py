import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert_absa')  # Path to your saved model
    tokenizer = BertTokenizer.from_pretrained('./fine_tuned_bert_absa')
    return model, tokenizer

model, tokenizer = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Streamlit app title
st.title("Aspect-Based Sentiment Analysis Web App")

# Input fields for sentence and aspect
sentence = st.text_area("Enter the review or sentence:")
aspect = st.text_input("Enter the aspect to analyze (e.g., 'food', 'service'):")

# Function to predict sentiment
def predict_sentiment(sentence, aspect):
    if not sentence or not aspect:
        return "Please enter both a sentence and an aspect."
    
    # Preprocess the input
    inputs = tokenizer.encode_plus(
        aspect,
        sentence,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map class index to sentiment
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[predicted_class]

# Button for prediction
if st.button("Analyze Sentiment"):
    result = predict_sentiment(sentence, aspect)
    st.subheader(f"Aspect: {aspect}")
    st.write(f"Predicted Sentiment: **{result}**")
