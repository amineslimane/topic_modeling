import streamlit as st
import pandas as pd
import numpy as np
from utils import *
import threading

df = pd.read_csv('dataset.csv', sep=",", index_col=None)
df_cleaned = pd.read_csv('dataset_cleaned.csv', sep=",", index_col=None)
df_cleaned.columns = ["📃 Texte", "⭐ Stars", "Length", "Cleaned Text"]


def index_input_callback():
    st.session_state['review'] = df.iloc[index_input]['text']


def aleatoire_callback():
    random_index = np.random.randint(df.shape[0], size=1)[0]
    st.session_state['index_input'] = random_index
    st.session_state['review'] = df.iloc[index_input]['text']

st.set_page_config(
    page_title="Review Analyzer | Topic Modeling",
    page_icon="⚡️",
    layout="wide",
)

with st.sidebar:
    st.title("Quel texte analyser ?")
    analyser_choice = st.radio("Quel texte analyser ?", ["Avis dataset", "Texte libre"])

    if analyser_choice == "Avis dataset":
        index_input = st.number_input("Numéro d'index", key="index_input", step=1, min_value=0, max_value=df.shape[0], on_change=index_input_callback)
        st.button("🤞🏼 Aléatoire", on_click=aleatoire_callback)

    page_css()
    page_js()

st.markdown("<h3>💬 Review Analyzer | Topic Modeling</h3>", unsafe_allow_html=True)
with st.expander("💡 Présentation du projet"):
    st.write("""
        L’intention de ce projet est de développer et mettre en œuvre des compétences de prétraitement de texte 
        et des techniques d’extraction de features spécifiques aux données non structurées de type texte dans le but 
        de détecter des sujets d’insatisfaction évoqués par des clients dans leurs avis postés sur les sites d’avis client.
        Le projet couvre tout le cycle de mise en place d’une preuve de concept, du prétraitement des données jusqu’au déploiement.
    """)
    etape_1, etape_2, etape_3, etape_4 = st.columns(4)
    etape_1.info("Etape 1️⃣ : Nettoyage et pré-traitement 🧹")
    etape_2.info("Etape 2️⃣ : Vectorisation et modélisation 🧠")
    etape_3.info("Etape 3️⃣ : Développement application web locale ✨")
    etape_4.info("Etape 4️⃣ : Déploiement application web 🚀")

with st.expander("📃 Données"):
    st.dataframe(df_cleaned.iloc[:, 0:2], height=250, use_container_width=True)

with st.expander("🚀 Code source"):
    with open('app.py', encoding="utf8") as f:
        st.code(f'{f.read()}')

review = st.text_area("Entrez un texte", height=150, max_chars=5000, key='review')
number = st.slider('Nombre de topics', value=3, step=1, min_value=1, max_value=15)


if review != "":
    detect_topic_btn = st.button(label="🤯 Détecter le sujet d'insatisfaction")

    if detect_topic_btn:
        t = threading.Thread(target=wait_spinner())
        t.start()
        suggested_topics = topics_suggestion(review, number)
        columns_components = st.columns(len(suggested_topics))
        i = 0
        for col in columns_components:
            col.metric(suggested_topics[i][0], suggested_topics[i][1])
            i += 1
        st.balloons()

        if len(suggested_topics) != number:
            st.warning(
                "⚠️ Le nombre de topic que vous avez demandé est supérieur au nombre de topic "
                "qui peuvent être en relation avec ce review (Probabilité de similarité égale à 0️%)"
            )