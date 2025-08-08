'''
💻 Sentiment Analysis Dashboard
This is a Streamlit web app that performs sentiment analysis on a dataset of text (e.g., reviews, comments, tweets) and provides:
📊 Visual analytics
☁️ Word clouds
🔍 Real-time sentiment prediction
📂 File upload support

📦 Features
✅ Analyze sentiments from a sample or custom dataset
✅ Classify text as Positive, Negative, or Neutral
✅ Visualize data using:

Pie chart

Bar chart

Word clouds (per sentiment)
✅ Real-time sentiment checker
✅ Upload your own CSV file (with a text column)

This script is used to run a sentiment analysis dashboard using Streamlit
To run this script, use the following command in your terminal:
streamlit run sentiment_dashboard.py
python -m streamlit run sentiment_dashboard.py
'''


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("sentiment_dataset.csv")
    return df

# Sentiment classification using TextBlob
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Generate WordCloud
def generate_wordcloud(texts):
    text_combined = " ".join(texts)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
    return wc

# --- Streamlit App ---
st.title("💻 Sentiment Analysis Dashboard")
st.markdown("Analyze text sentiments using NLP. Real-time predictions, visualizations & more!")

# Load and process data
df = load_data()
df['Sentiment'] = df['text'].apply(get_sentiment)

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment to View Comments", 
    options=["Positive", "Neutral", "Negative"], 
    default=["Positive", "Neutral", "Negative"]
)

# Display sentiment distribution
st.subheader("📊 Sentiment Distribution")

col1, col2 = st.columns(2)

with col1:
    sentiment_counts = df['Sentiment'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax2, color=["green", "gray", "red"])
    ax2.set_ylabel("Number of Texts")
    ax2.set_title("Sentiment Counts")
    st.pyplot(fig2)

# Display filtered comments
st.subheader("📝 Sample Comments by Sentiment")
for sentiment in sentiment_filter:
    st.markdown(f"### {sentiment}")
    samples = df[df['Sentiment'] == sentiment].sample(min(3, len(df[df['Sentiment'] == sentiment])))
    for i, row in samples.iterrows():
        st.write(f"• {row['text']}")

# Word Clouds
st.subheader("☁️ Word Clouds by Sentiment")
cols = st.columns(3)

for idx, sentiment in enumerate(["Positive", "Neutral", "Negative"]):
    texts = df[df['Sentiment'] == sentiment]['text']
    if not texts.empty:
        wordcloud = generate_wordcloud(texts)
        cols[idx].markdown(f"**{sentiment}**")
        cols[idx].image(wordcloud.to_array(), use_container_width=True)

# Real-time sentiment checker
st.subheader("🔍 Real-Time Sentiment Checker")
user_input = st.text_input("Enter a sentence to analyze:")
if user_input:
    sentiment = get_sentiment(user_input)
    st.markdown(f"**Sentiment:** :{'smiley' if sentiment=='Positive' else 'neutral_face' if sentiment=='Neutral' else 'disappointed'}: `{sentiment}`")

# Optional: Upload custom file
st.sidebar.header("📂 Upload Custom Dataset (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with 'text' column")

if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    if 'text' in user_df.columns:
        user_df['Sentiment'] = user_df['text'].apply(get_sentiment)
        st.sidebar.success("File processed successfully!")
        st.write("### Uploaded Data Sentiment Preview", user_df.head())
    else:
        st.sidebar.error("CSV must have a column named 'text'.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & TextBlob | AI Intern – Task 7")
