import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pickle
import http

nltk.download('punkt', quiet = True)
nltk.download('stopwords', quiet = True)

# PorterStemmer object initiate
ps = PorterStemmer()

def transform_text(text):
    # lower casing
    text = text.lower()
    # converting text into list of words
    text = nltk.word_tokenize(text)

    y = []
    # removing special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # removing stopwords/helping words
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Normalization of word i.e converting words into their base form.
    for j in text:
        y.append(ps.stem(j))

    return " ".join(y)


tfidf = pd.read_pickle('models/vectorizer.pkl')
model = pd.read_pickle('models/model.pkl')

# with open("models/model.pkl", "rb") as f:
#     object = pickle.load(f)
    
# df = pd.DataFrame(object)
# df.to_csv(r'model.csv')


# df = pd.read_csv('models/url_spam_classification.csv')
# df.to_pickle('url_spam_classification.pkl')
# tfidf2 = pd.read_pickle('url_spam_classification.pkl')
# pickle.dump(model,open('models/urlML.py','wb'))
# model2 = pd.read_pickle('model2.pkl')

st.title(':rainbow[*Cyber Kavach Spam Detector*]')
st.markdown("-------------------")
st.markdown('##### Discover if your text messages or mails or urls are safe or sneaky! Try this Cyber Kavach Spam Detector now!')
# st.markdown('###### This model can detect spam messages with an accuracy of 97%.')

st.markdown(" ")
user_input = st.text_input('Enter text here')

st.markdown(" ")
url_input = st.text_input('Enter URL here')

st.markdown(" ")
exe_input = st.text_input('Enter exe file path here')

if st.button("Predict"):
    # if user_input[:] == "":
    #     st.warning("Please enter a message.")
    if user_input[:]:
        # Preprocess user input
        transformed_txt = transform_text(user_input)
        converted_num = tfidf.transform([transformed_txt])
        result = model.predict(converted_num)[0]

        # Display prediction
        if result == 1:
            st.error("SPAM")
        else:
            st.success("Not Spam")
    elif url_input[:]:
        # Preprocess user input
        transformed_txt = transform_text(url_input)
        converted_num = tfidf.transform([transformed_txt])
        result = model.predict(converted_num)[0]
        

        # Display prediction
        if result == 1:
            st.error("SPAM")
        else:
            st.success("Not Spam")
    else:
        # Preprocess user input
        transformed_txt = transform_text(exe_input)
        converted_num = tfidf.transform([transformed_txt])
        result = model.predict(converted_num)[0]

        # Display prediction
        if result == 1:
            st.error("SPAM")
        else:
            st.success("Not Spam")