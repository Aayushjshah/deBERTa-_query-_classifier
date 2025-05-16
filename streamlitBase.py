import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
import re
from nltk.corpus import stopwords
from textblob import TextBlob
import numpy as np
# Download necessary NLTK data
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Configuration
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 32
device = "cuda" if torch.cuda.is_available() else "cpu"


from huggingface_hub import login

# login( #Use you api key


# load Model 
try:
    model = AutoModelForSequenceClassification.from_pretrained('./model/').to('cpu') # Or 'cpu'

    # 2.  Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
except:
    print("error in lading model")


# Load model and tokenizer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
def get_prediction(text, max_len, device='cpu',gen_task_query=False):
    """
    Predicts the sentiment of a given text using the trained model.

    Args:
        text (str): The input text to predict the sentiment for.
        model (torch.nn.Module): The trained PyTorch model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for the model.
        max_len (int): The maximum sequence length.
        device (str, optional): The device to use ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        dict: A dictionary containing the predicted sentiment ('Positive' or 'Negative')
              and its corresponding probability.
    """
    user_query = text
    text = preprocess_query(text)  # Apply the preprocessing here
    model.eval()  # Set the model to evaluation mode
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**encoding)
        logits = outputs.logits

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu()).numpy()  # Get probabilities, move to CPU, convert to numpy

    label = np.argmax(probs, axis=-1)
    task_query = ""
    # gen_task_query = True
    # if(gen_task_query == True):
    #     if label == 1:
    #         task_query = generate_calendar_search_query(user_query)
    #         task_query = extract_after_last_search_query(task_query)
    #     else:
    #         task_query = generate_email_search_query(user_query)
    #         task_query = extract_after_last_search_query(task_query)
    
    if label == 1:
        return {
            'prediction': 'Calendar',
            'probability': probs[1],
            'task_query' : task_query
        }
    else:
        return {
            'prediction': 'Email',
            'probability': probs[0],
            'task_query' : task_query
        }
    
    
    
# Preprocessing function
def preprocess_query(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = str(TextBlob(text).correct())
    text = re.sub(r"[^\w\s@.]", "", text)
    default_stop_words = set(stopwords.words("english"))
    retain_words = {
        "what", "how", "when", "where", "who", "which", "whom", "whose", "why",
        "can", "should", "would", "could", "do", "did", "does", "will", "may",
        "show", "find", "search", "get", "have"
    }
    custom_stop_words = default_stop_words - retain_words
    words = text.split()
    filtered_words = [word for word in words if word not in custom_stop_words]
    text = " ".join(filtered_words).strip()
    return text

# Main Streamlit app
def main():
    st.title("Query Classification and Task Query Generation")
    st.write("Enter your query to get a classification and a generated task query.")

    user_query = st.text_input("Your Query:", "")
    gen_task_query = st.checkbox("Generate Task Query", value=True) # added a checkbox

    if st.button("Submit"):
        if not user_query:
            st.error("Please enter a query.")
            return

        # Get prediction
        prediction = get_prediction(user_query, MAX_LEN, 'cpu')
        print(prediction['task_query'] , "AA")
        st.write(f"**Predicted Label:** {prediction['prediction']}")
        st.write(f"**Generated Task Query:** Task query generation is disabled.")
        # if gen_task_query: # Only generate if the box is checked.
        #     # Generate task query
        #     task_query = generate_task_query(user_query)
        #     st.write(f"**Generated Task Query:** {task_query}")
        # else:
            

if __name__ == "__main__":
    main()
