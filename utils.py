import numpy as np
import pickle
import time
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


topics = [
    '👨 Staff management',
    '🕘 Waiting Time',
    '🍕 Pizza',
    '🛎️ Customer Service',
    '👨‍🍳 Food Quality',
    '🍔 Burger',
    '🍴 Ordering & Delivery to table',
    '🌍 Place Environnement',
    '🐔 Menu Chicken',
    '🥤 Drinks',
    'Experience',
    '🌍 Location',
    '😵 Taste',
    '🍣 Sushi and Rice',
    '🥪 Sandwich']

model = pickle.load(open('nmf_model/model', 'rb'))
vectorizer = pickle.load(open('nmf_model/vectorizer', 'rb'))


def review_is_positive(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    if sentiment_dict['compound'] >= 0.4:
        return True
    return False


def topics_suggestion(text, nb):
    transformed_text = vectorizer.transform([text])
    predicted_topics = model.transform(transformed_text)
    sorted_predicted_topics = np.argsort(predicted_topics, axis=1)
    final_predicted_topics = []
    for i in range(len(predicted_topics)):
        for j in range(len(topics) - 1, len(topics) - 1 - nb, -1):
            topic_index = sorted_predicted_topics[i][j]
            topic = topics[topic_index]
            topic_percentage = round(100*predicted_topics[i][topic_index], 1)
            if topic_percentage == 0:
                break
            final_predicted_topics.append([topic, str(topic_percentage)+"%"])
    return final_predicted_topics


def wait_spinner():
    with st.spinner('⏳ Wait for it...'):
        time.sleep(5)
    st.success('Success!')
