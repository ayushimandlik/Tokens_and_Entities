import streamlit as st
from transformers import AutoTokenizer, BertForTokenClassification
from utils import tokenize_and_predict
import torch

# Load model and tokenizer
model = BertForTokenClassification.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")

st.title("Named Entity Recognition with BERT")
sentence = st.text_input("Enter a sentence to analyze:")

if sentence:
    predictions = tokenize_and_predict(sentence, tokenizer, model)
    #tokens,predictions = tokenize_and_predict(sentence, tokenizer, model)
    st.markdown("### Token-wise NER Tags")
    for prediction in predictions:
        st.write(prediction)
   # for token, label in zip(tokens, predictions):
   #     st.write(f"`{token}` ➡️ **{label}**")

