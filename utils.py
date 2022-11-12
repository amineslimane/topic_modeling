import numpy as np
import pickle
import time
import streamlit as st


uploaded_pickled_model = pickle.load(open('nmf_model/model', 'rb'))
model_vectorizer = pickle.load(open('nmf_model/vectorizer', 'rb'))
topics = ['Staff management', '👨‍🍳 Food Quality', '🍕 Pizza', '🐔 Menu Chicken', '🥣Quality', '⏱ Service time',
           '🍔 Burger', '🕘 Waiting Time', '🖐 Experience', '🥤 Drinks', '🍴 Ordering & Delivery to table', '🌍 Location',
           '🛎️ Customer Service',  '🍣 Sushi and Rice', '🌍 Place Environnement']


def topics_suggestion(text, nb):
    transformed_text = model_vectorizer.transform([text])
    predicted_topics = uploaded_pickled_model.transform(transformed_text)
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
