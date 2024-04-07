import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the dataset
data = pd.read_csv('newdataset.csv')

# Fill NaN values in the 'text' column
data['text'].fillna('', inplace=True)

# Load the pre-trained classifier
with open('C:/Users/palla/OneDrive/Desktop/rumor_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(data['text'])

st.title('Rumor classification')
st.subheader('Enter Rumor :')
rumor = st.text_input('Rumor', '')
if st.button('CLASSIFY'):
    if rumor:
        rum_vect = vectorizer.transform([rumor])
        prediction = classifier.predict(rum_vect)
        st.subheader('Prediction:')
        st.write(prediction[0])
    else:
        st.warning('Please enter a rumor...')
