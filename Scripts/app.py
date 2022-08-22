import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

def transform_text(text):
    '''
    This functions performs data preprocessing on text
    
    Parameters :
        text :: str
            string to be transformed
    
    Returns : 
        new_text :: str
            transformed string
    '''
    
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    #Removing special characters
    new_text =  []
    for alpha in text:
        if alpha.isalnum():
            new_text.append(alpha)
    
    text = new_text[:]
    new_text.clear()
    
    #Removing stopwords and punctuations
    for alpha in text:
        if alpha not in stopwords.words('english') and alpha not in string.punctuation:
            new_text.append(alpha)
   
    #Stemming
    ps = PorterStemmer()
    text = new_text[:]
    new_text.clear()
    
    for alpha in text:
        new_text.append(ps.stem(alpha))
    
    new_text = " ".join(new_text)
    return new_text


model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title('SMS Spam Classifier')
input_text = st.text_area("Enter the message")
#Our pipeline should be as follows
#Preprocess
#Vectorize
#Predict

#Preprocess
transformed_text = transform_text(input_text)

#Vectorize
vector = vectorizer.transform([transformed_text])

#Predict
result = model.predict(vector)[0]


if st.button("Predict"):
	if result == 1:
		st.header("Spam")
	else: 
		st.header("Not Spam")