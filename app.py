import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Loading tokenizer and model (caching to prevent reloading every time)
@st.cache_resource()
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert_toxic_model/')
    model = BertForSequenceClassification.from_pretrained('bert_toxic_model/')
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Class labels
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Streamlit UI
st.title("Toxic Comment Classifier (BERT)")

user_input = st.text_area("Enter a comment to check for toxicity:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a comment")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

        # Displaying results
        st.subheader("Prediction Probabilities:")
        for label, prob in zip(class_names, probs):
            st.write(f"**{label}**: {prob:.4f}")
