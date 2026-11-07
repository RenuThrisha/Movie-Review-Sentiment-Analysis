from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import streamlit as st

model=load_model('rnn_imdb.h5')

word_index=imdb.get_word_index()
reversed_word_index={value:key for key,value in word_index.items()}

def preprocessing(review):
    review=review.lower().split()
    encoded=[word_index.get(word,2)+3 for word in review]
    padded_review=pad_sequences([encoded],maxlen=500)
    return padded_review

def predict_review(review):
    preprocessed_review=preprocessing(review)
    prediction=model.predict(preprocessed_review)
    sentiment='positive' if prediction[0][0]>0.5  else 'negative'
    return sentiment,prediction[0][0]


st.title("IMDB sent analysis")
st.write("enter you review ")

user_input=st.text_area('Movie revierw')

if st.button('Classify'):
    sentiment,score=predict_review(user_input)
    st.write(sentiment)
    st.write(score)
else:
    st.write("pls write the review")












