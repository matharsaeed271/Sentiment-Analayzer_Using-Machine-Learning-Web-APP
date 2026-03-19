from textblob import TextBlob
import pandas as pd
import streamlit as st
from cleantext import clean
import emoji
from textblob import TextBlob
import nltk

import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#################################
# analyzer = SentimentIntensityAnalyzer()

# text = st.text_input("Enter your text:")

# if text:
#     score = analyzer.polarity_scores(text)
    
#     st.write(score)

# bad_words = ["shut up", "get out", "stupid", "idiot", "fuck", "fucked", "hell", "pig"]

# def custom_sentiment(text):
#     for word in bad_words:
#         if word in text.lower():
#             return "Negative 😡"
    
#     score = analyzer.polarity_scores(text)
    
#     if score['compound'] < 0:
#         return "Negative 😡"
#     elif score['compound'] > 0:
#         return "Positive 😊"
#     else:
#         return "Neutral 😐"

############################################
nltk.download('punkt')
nltk.download('stopwords')
st.title("Sentiment-Web-Analyzer")
background_image = '1752066186248.jpg'
st.image(background_image, width=700)

st.header("Scale Your Thoughts")

with st.expander("Analyze Your Text"):
    text = st.text_input("Text here:")

    if text:
        blob = TextBlob(text)
        p= round(blob.sentiment.polarity,2)
        st.write('Polarity :',p)
        if p>=0.1:
               st.write(emoji.emojize("Positive Speech :grinning_face_with_big_eyes:"))
        elif p==0.0:
            st.write(emoji.emojize("Neutral Speech :zipper-mouth_face:"))
        else :
            st.write(emoji.emojize("Negative Speech :disappointed_face:"))
        st.write('Subjectivity', round(blob.sentiment.subjectivity,2))

    pre = st.text_input('Clean Your Text: ')
    if pre:
        cleaned_text = clean(
            pre,
            clean_all=False,
            extra_spaces=True,
            stopwords=True,
            lowercase=True,
            numbers=True
        )
        st.write(cleaned_text)

### yahan sy
import transformers
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

st.text("If you don't satisfy from above analyzer (optional)")
text = st.text_input("Enter your text:")

if text:
    result = model(text)[0]

    label = result['label']
    score = result['score']

    # Convert to polarity
    if label == "POSITIVE":
        polarity = score
    else:
        polarity = -score

    # Subjectivity (approximation)
    subjectivity = abs(polarity)

    # Speech label
    if polarity > 0:
        speech = "Positive 😊"
    elif polarity < 0:
        speech = "Negative 😡"
    else:
        speech = "Neutral 🤐"

    st.write(f"Polarity : {round(polarity,2)}")
    st.write(f"{speech} Speech")
    st.write(f"Subjectivity : {round(subjectivity,2)}")

    ### yahan tak

with st.expander('Analyze Excel files'):
    st.write("_**Note**_ : Your file must contain the column Name'Tweets' that contain the text to be analyzed.")
    upl = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:
        df = pd.read_excels(upl)
        # del df['Unnamed: 0']
        df['score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(10))

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
st.write("\n" * 15)
# Add a bold line above the footer
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
# Footer content
st.write("Copy© 2026 M.Athar | Made With Muhammad Athar Ur Rahman")
