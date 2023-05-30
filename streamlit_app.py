import streamlit as st

import text_preprocessing
import load_model

# title
st.title("Analyze Your Tweet Sentiment!!!")

# load model
with st.spinner("Loading Model...."):
    my_model = load_model.load_model()

# get new tweet
tweet = st.text_area("What's happening? (Indonesian tweet only)","")

if st.button("Send", use_container_width=True):
    with st.spinner("Please wait, analyzing tweet sentiment..."):
        df, preprocess_text = load_model.predict(my_model, tweet)
        st.success(df.sentiment.iloc[0])
    
    st.divider() # draw a horizontal line
        
    # show table of all categories
    with st.expander("Table all of sentiment"):
        st.dataframe(df, use_container_width=True)
