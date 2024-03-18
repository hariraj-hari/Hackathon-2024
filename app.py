import pandas as pd
import streamlit as st
import pickle
import re
import string
from nltk import word_tokenize as wtk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity



import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

scaler = MaxAbsScaler()
tfidf_vectorizer = TfidfVectorizer()
lemmatizer = WordNetLemmatizer()

regex_pattern = r'[()>>\r\n\r-----------------------\0-9x x x x x X X X X X]'

def clean_subject_content_text(text):
    cleaned_text = re.sub(regex_pattern, ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text
    
def remove_stopwords_text(text):
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(cleaned_words)

def remove_punctuation(text):
    # Define the punctuation marks to remove
    punctuation_marks = string.punctuation
    # Remove punctuation marks
    text_without_punctuation = ''.join([char for char in text if char not in punctuation_marks])
    return text_without_punctuation

# Load the data
df_data = pickle.load(open('raw_data_set.pkl', 'rb'))
df =  pd.DataFrame(df_data)
df_action = pd.read_json("no_pii_action_history.json")
st.header('Auto Clustering System')


# Input processing
user_input = st.text_input("Enter Your Grievances")

# Check if user input is not empty
if user_input:
    cleaned_text = clean_subject_content_text(user_input)
    cleaned_text = remove_punctuation(cleaned_text)
    cleaned_text_tokens = wtk(cleaned_text)
    cleaned_text = ' '.join([word.lower() for word in cleaned_text_tokens])  
    cleaned_text = remove_stopwords_text(cleaned_text) 
    cleaned_text = lemmatizer.lemmatize(cleaned_text) 

    # Check if cleaned text is not empty
    if cleaned_text:
        tfidf_matrix_cleaned_text = tfidf_vectorizer.fit_transform([cleaned_text])
        tfidf_matrix_cleaned_text_scalar = scaler.fit_transform(tfidf_matrix_cleaned_text)
        cos_similarities = cosine_similarity(tfidf_matrix_cleaned_text_scalar, tfidf_matrix_cleaned_text)
        N:int = 10
        top_similar_indices = cos_similarities.argsort()[0][-N:][::-1]
        top_similar_texts = df.iloc[top_similar_indices]
        df_res =top_similar_texts
        merged_df = pd.merge(df_action, df_res, on='registration_no', how='left')
        merged_df = merged_df.dropna(subset=['OfficerDetail'])
        columns = st.columns(2) 
        with columns[0]:
            st.subheader("Officer Detail")
            st.write(merged_df['OfficerDetail'].iloc[0])
            st.write(merged_df['OfficerDetail'].iloc[1])
        with columns[1]:
            st.subheader("Organization Code")
            st.write(merged_df['org_code_x'].iloc[0])
            st.write(merged_df['org_code_x'].iloc[1])
    else:
        st.write("Input text after preprocessing is empty. Please provide meaningful text.")
else:
    st.write("Please enter some text.")
