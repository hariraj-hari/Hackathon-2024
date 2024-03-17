import streamlit as st
import pandas as pd
import pickle
import re
import string
import nltk
from nltk import word_tokenize as wtk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
df = pickle.load(open('raw_data_set.pkl', 'rb'))
df_action = pd.read_json("C:\\Users\\aravi\\OneDrive\\Documents\\problem_statement_1_and_2\\no_pii_action_history\\no_pii_action_history.json")

# Clean text functions
def clean_subject_content_text(text):
    cleaned_text = re.sub(r'[()>>\r\n\r-----------------------\0-9x x x x x X X X X X]', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def remove_punctuation(text):
    punctuation_marks = string.punctuation
    text_without_punctuation = ''.join([char for char in text if char not in punctuation_marks])
    return text_without_punctuation

def remove_stopwords_text(word_list):
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word for word in word_list if word.lower() not in stop_words]
    return cleaned_words

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif nltk_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# TF-IDF Vectorization and scaling
tfidf_matrix = vectorizer.fit_transform(df['modified_text'])
scaler = MaxAbsScaler()
tfidf_matrix_scaled = scaler.fit_transform(tfidf_matrix)

# Streamlit app
def main():
    st.header('Auto Clustering System')
    # Get text input from the user
    input_text = st.text_input("Enter some text:")
    
    # Check if the user has entered any text
    if input_text:
        # Clean and preprocess the input text
        cleaned_text = clean_subject_content_text(input_text)
        cleaned_text = remove_punctuation(cleaned_text)
        cleaned_text = wtk(cleaned_text)
        cleaned_text = remove_stopwords_text(cleaned_text)
        cleaned_text = [lemmatizer.lemmatize(word) for word in cleaned_text]
        cleaned_text = ' '.join(cleaned_text)
        
        # TF-IDF Vectorization and scaling of the input text
        input_vector = vectorizer.transform([cleaned_text])
        input_vector_scaled = scaler.transform(input_vector)
        
        # Compute cosine similarities between input text and the dataset
        cos_similarities = cosine_similarity(input_vector_scaled, tfidf_matrix_scaled)
        
        # Get the index of the top similar texts
        N = 10  # Adjust as needed
        top_similar_indices = cos_similarities.argsort()[0][-N:][::-1]
        
        # Retrieve the top similar texts from the DataFrame based on the indices
        top_similar_texts = df.iloc[top_similar_indices]
        
        # Merge with action history data
        merged_df = pd.merge(df_action, top_similar_texts, on='registration_no', how='left')
        
        # Filter out rows where 'OfficerDetail' is not None
        merged_df = merged_df.dropna(subset=['OfficerDetail'])
        
        # Display officer details
        if not merged_df.empty:
            st.write("Redirect complaint to the:", merged_df['OfficerDetail'].iloc[0])
        else:
            st.write("No matching records found.")

if __name__ == "__main__":
    main()
