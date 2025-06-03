import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import streamlit as st
import pickle as pkl

ps=PorterStemmer()
tfidf=pickle.load(open("vectorizer.pkl","rb"))
model=pickle.load(open("model.pkl","rb"))
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    words = y[:]
    y.clear()

    return " ".join([ps.stem(word) for word in words])



st.title("Spam Classifier")
input_sms=st.text_area("Enter the message.")
if st.button('Predict'):
    # preprocess
    transformed_text = transform_text(input_sms)
    # vectorize
    vector = tfidf.transform([transformed_text])
    # predict
    result = model.predict(vector)
    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header('Not Spam')

