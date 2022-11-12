import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from utils import topics_suggestion, wait_spinner, review_is_positive
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
    page_icon="⚡️",
    layout="wide",
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
    # "https://colab.research.google.com/github/amineslimane/topic_modeling/blob/master/app.py"
    open_in_github_svg = open('static/images/open_in_github.svg', 'r', encoding="utf8").read()
    open_in_colab_svg = open('static/images/open_in_colab.svg', 'r', encoding="utf8").read()
    open_in_kaggle = open('static/images/open_in_kaggle.svg', 'r', encoding="utf8").read()
    st.markdown(
        """
            <a href='https://github.com/amineslimane/topic_modeling' style='display: inline-block; margin-right:5px; margin-bottom:10px; color: white;'>
                {}</svg>
            </a>
            <a href='#' style='display: inline-block; margin-right:5px; margin-bottom:10px; color: white;'>
                {}</svg>
            </a>
            <a href='#' style='display: inline-block; margin-right:5px; margin-bottom:10px; color: white;'>
                {}</svg>
            </a>
        """.format(open_in_github_svg, open_in_colab_svg, open_in_kaggle),
        unsafe_allow_html=True)

    code_tab1, code_tab2, code_tab3 = st.tabs(["📃 app.py", "📃 Cleaned Dataset", "📃 Negative Dataset"])

    with code_tab1:
        st.header("📃 app.py")
        code_file1 = open('app.py', 'r', encoding="utf8").read()
        st.download_button('Download', code_file1, file_name="app.py")
        st.code(f'{code_file1}')

    with code_tab2:
        st.header("📃 preprocessing.py")
        code_file2 = open('machine_learning/preprocessing.py', 'r', encoding="utf8").read()
        st.download_button('Download', code_file2, file_name="preprocessing.py")
        st.code(f'{code_file2}')

    with code_tab3:
        st.header("📃 build_model.py")
        code_tab3 = open('machine_learning/build_model.py', 'r', encoding="utf8").read()
        st.download_button('Download', code_tab3, file_name="build_model.py")
        st.code(f'{code_tab3}')


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



# options = st.multiselect(
#     'What are your favorite colors',
#     ['Red', 'Blue'],)
#
# st.write('You selected:', options)
# import time
# import requests
#
# from streamlit_lottie import st_lottie
# from streamlit_lottie import st_lottie_spinner
#
#
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()
#
#
# lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
# lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
# lottie_hello = load_lottieurl(lottie_url_hello)
# lottie_download = load_lottieurl(lottie_url_download)
#
# lottie_url_404 = "https://labs.nearpod.com/bodymovin/demo/markus/isometric/markus2.json"
# lottie_404 = load_lottieurl(lottie_url_404)
# st_lottie(lottie_404, width=500, key="404")
#
#
# st_lottie(lottie_hello, width=200, key="hello")
#
# if st.button("Download"):
#     with st_lottie_spinner(lottie_download, key="download"):
#         time.sleep(5)
#     st.balloons()

# st.info("This project application helps you build and explore your data.")

# def update_slider():
#     st.session_state.slider = st.session_state.numeric
#
# def update_numin():
#     st.session_state.numeric = st.session_state.slider
#
# val = st.number_input('Input', value=0, key='numeric', on_change=update_slider)
#
# slider_value = st.slider('slider', min_value=0,
#                          value=val,
#                          max_value=5,
#                          step=1,
#                          key='slider', on_change=update_numin)


# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
#
# def remote_css(url):
#     st.markdown('<style src="{}"></style>'.format(url), unsafe_allow_html=True)
#
# def icon_css(icone_name):
#     remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
#
# def icon(icon_name):
#     st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)

# local_css('style.css')
# html_string = "<h3>this is an html string</h3><style>body{background-color:red}</style>"
# st.markdown(html_string, unsafe_allow_html=True)
# components.html("""
#     <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
#     <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
#     <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
#     <style>body{background-color:red}</style>
#     <div id="accordion">
#       <div class="card">
#         <div class="card-header" id="headingOne">
#           <h5 class="mb-0">
#             <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
#             Collapsible Group Item #1
#             </button>
#           </h5>
#         </div>
#         <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
#           <div class="card-body">
#             Collapsible Group Item #1 content
#           </div>
#         </div>
#       </div>
#       <div class="card">
#         <div class="card-header" id="headingTwo">
#           <h5 class="mb-0">
#             <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
#             Collapsible Group Item #2
#             </button>
#           </h5>
#         </div>
#         <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
#           <div class="card-body">
#             Collapsible Group Item #2 content
#           </div>
#         </div>
#       </div>
#     </div>
#     """,height=600,)