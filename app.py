import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from utils import topics_suggestion, wait_spinner, review_is_positive, show_code, open_in_github2_svg
import threading
import streamlit.components.v1 as components
import altair as alt
import time


df = pd.read_csv('data/dataset.csv', sep=",", index_col=None)
df_cleaned = pd.read_csv('data/dataset_cleaned.csv', sep=",", index_col=None)
df_negative = pd.read_csv('data/dataset_negative.csv', sep=",", index_col=None)

df.columns = ["📃 Text", "⭐ Stars"]
df_cleaned.columns = ["📃 Text", "⭐ Stars", "Length", "🧼 Cleaned Text"]
df_negative.columns = ["📃 Text", "⭐ Stars", "Length", "🧼 Cleaned Text"]


def index_input_callback():
    st.session_state['review'] = df_negative.iloc[index_input]['📃 Text']


def aleatoire_callback():
    random_index = np.random.randint(df_negative.shape[0], size=1)[0]
    st.session_state['index_input'] = random_index
    st.session_state['review'] = df_negative.iloc[index_input]['📃 Text']


# im = Image.open("favicon.ico")
st.set_page_config(
    page_title="Review Analyzer | Topic Modeling",
    page_icon="💬️",
    layout="wide",
    # initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("Quel texte analyser ?")
    analyser_choice = st.radio("Quel texte analyser ?", ["Avis dataset", "Texte libre"])

    if analyser_choice == "Avis dataset":
        index_input = st.number_input("Numéro d'index", key="index_input", step=1, min_value=0, max_value=df.shape[0], on_change=index_input_callback)
        st.button("🤞🏼 Aléatoire", on_click=aleatoire_callback)

    with open('static/style.css') as f:
        css_component = f'<style>{f.read()}</style>'
    st.markdown(css_component, unsafe_allow_html=True)

    with open('static/main.js') as f:
        javascript_component = f'<script>{f.read()}</script>'
    components.html(javascript_component, height=0)

# Main Content

st.markdown(
    """
        <h3>💬 Review Analyzer | Topic Modeling
            <a href='https://github.com/amineslimane/topic_modeling' style='display: inline-block; margin-left:10px; margin-bottom:10px; color: white;'>
                {}</svg>
            </a>
        </h3>
    """.format(open_in_github2_svg),
    unsafe_allow_html=True)

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

    col_img_1, col_img_2 = st.columns(2)
    image_negative_words = Image.open('static/images/most frequent negative words.png')
    image_positive_words = Image.open('static/images/most frequent positive words.png')
    with col_img_1:
        st.image(image_negative_words, caption='Most frequent negative words')
    with col_img_2:
        st.image(image_positive_words, caption='Most frequent positive words')

with st.expander("🧊 Données"):
    tab1, tab2, tab3 = st.tabs(["📃 Dataset", "🧼 Cleaned Dataset", "👎 Negative Dataset"])

    with tab1:
        st.header("📃 Dataset")
        data_file = open('data/dataset.csv', 'r', encoding="utf8").read()
        st.download_button('Download dataset', data_file, file_name="dataset.csv")
        st.dataframe(df, height=250, use_container_width=True)

    with tab2:
        st.header("🧼 Cleaned Dataset")
        data_file = open('data/dataset_cleaned.csv', 'r', encoding="utf8").read()
        st.download_button('Download cleaned data', data_file, file_name="dataset_cleaned.csv")
        st.dataframe(df_cleaned, height=250, use_container_width=True)

    with tab3:
        st.header("👎 Negative Dataset")
        data_file = open('data/dataset_negative.csv', 'r', encoding="utf8").read()
        st.download_button('Download negative data', data_file, file_name="dataset_negative.csv")
        st.dataframe(df_negative, height=250, use_container_width=True)

with st.expander("🚀 Code source"):
    code_tab1, code_tab2, code_tab3 = st.tabs(["📃 app.py", "✂ preprocessing.py", "🔥 build_model.py"])
    with code_tab1:
        show_code("app.py", "📃 ")
    with code_tab2:
        show_code("preprocessing.py", "✂ ")
    with code_tab3:
        show_code("build_model.py", "🔥 ")


review = st.text_area("Entrez un texte", height=150, max_chars=5000, key='review')
number = st.slider('Nombre de topics', value=3, step=1, min_value=1, max_value=15)

if review != "":
    detect_topic_btn = st.button(label="🤯 Détecter le sujet d'insatisfaction")

    if detect_topic_btn:
        if review_is_positive(review):
            st.warning("This review is positive, write negative opinion to detect the topic")
        else:
            t = threading.Thread(target=wait_spinner())
            t.start()
            suggested_topics = topics_suggestion(review, number)
            columns_components = st.columns(len(suggested_topics))

            topics = [x[0] for x in suggested_topics]
            probabilities = [float(x[1].replace("%", ""))/100 for x in suggested_topics]
            topic_probability_combined = [" ".join(x) for x in suggested_topics]

            source = pd.DataFrame({
                'Probabilité': probabilities,
                'Topic': topics,
                'Topic ': topic_probability_combined
            })


            bar_chart = alt.Chart(source,title="Topic Modeling").mark_bar(color='#03045e').encode(
                y='Probabilité',
                x='Topic',
                tooltip=["Topic "],
            ).interactive()
            st.altair_chart(bar_chart, use_container_width=True)

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
st.markdown("""
    <footer class="css-qri22k">
        Made by 
        <a href="https://amineslimane.me" class="css-1vbd788 egzxvld1">
            <img src="https://amineslimane.me/images/photo.png" style="width: 20px; margin: 0px 3px 4px 4px;">
            Amine Slimane
        </a>
        
    </footer>
""", unsafe_allow_html=True)