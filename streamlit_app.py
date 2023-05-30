import streamlit as st

import text_preprocessing
import model

# title
st.title("Analyze Your Tweet Sentiment!!!")

# load model
with st.spinner("Loading Model...."):
    my_model = model.load_model()

# get new tweet
tweet = st.text_area("What's happening? (Indonesian tweet only)","")

if st.button("Send", use_container_width=True):
    with st.spinner("Please wait, analyzing tweet sentiment..."):
        df, preprocess_text = model.predict(my_model, tweet)
        for cat, prob in zip(df['sentiment'], df['probability']):
            if prob < 5:
                break
            pred = cat + " | " + str(round(prob,2)) + "%"
            st.success(pred)
    
    st.divider() # draw a horizontal line
        
    # show table of all categories
    with st.expander("Table all of sentiment"):
        st.dataframe(df, use_container_width=True)
