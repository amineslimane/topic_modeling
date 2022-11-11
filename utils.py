import numpy as np
import pickle
import time
import streamlit as st
import streamlit.components.v1 as components

uploaded_pickled_model = pickle.load(open('nmf_model/model_yasmine', 'rb'))
model_vectorizer = pickle.load(open('nmf_model/vectorizer_yasmine', 'rb'))
topics = ['Staff management', '👨‍🍳 Food Quality', '🍕 Pizza', '🐔 Menu Chicken', '🥣 Quality', '⏱ Service time',
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


def page_css():
    with open('static/style.css') as f:
        css_component = f'<style>{f.read()}</style>'
    st.markdown(css_component, unsafe_allow_html = True)


def page_js():
    with open('static/main.js') as f:
        javascript_component = f'<script>{f.read()}</script>'
    components.html(javascript_component, height=0)