import gradio as gr
import joblib

import re
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

#some data required for preprocessing
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")


#Load trained model and tf-idf vectorizer
rf_model = joblib.load('random_forest_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')


#Functions related to preprocessing
def remove_pfx(url):
    url = re.sub(r'^www\.', '', url)
    return url

def Tokenize(url):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    url=tokenizer.tokenize(url)
    return url

def remove_stopwords(tokenized_text):
    filtered_text = [word for word in tokenized_text if word.lower() not in stop_words]
    return filtered_text

def snow_stem(t):
    t=[stemmer.stem(word) for word in t]
    t=' '.join(t)
    return t

def preprocess(url):
    url=snow_stem(remove_stopwords(Tokenize(remove_pfx(url))))
    feature = tfidf_vectorizer.transform([url])
    return feature

#Classifier Function (bridge between input and output)
def classify_url(url):
    prediction = rf_model.predict(preprocess(url))
    if prediction[0] == 0:
        return "This URL appears to be BENIGN"
    else:
        return "MALICIOUS URL , Be Careful !"


#Gradio User-Interface 
url_input = gr.Textbox(label="Enter URL:")
label = gr.Textbox(label="Prediction")

#Passing the classifier function and launching the interface
gr.Interface(fn=classify_url, inputs=url_input, outputs=label, title="Malicious URL Classifier",allow_flagging="never",theme="reilnuud/polite").launch()
