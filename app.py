from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
import numpy as np
from streamlit_pandas_profiling import st_profile_report
import os 
import datetime as dt
from PIL import Image
import streamlit.components.v1 as components


df = pd.read_csv('dataset.csv', sep=",", index_col=None)

def index_input_callback():
    st.session_state['options'] = df.iloc[index_input]['text']

def aleatoire_callback():
    random_index = np.random.randint(df.shape[0], size=1)[0]
    st.session_state['index_input'] = random_index
    st.session_state['options'] = df.iloc[index_input]['text']


st.set_page_config(
    page_title="Review Analyzer | Topic Modeling",
    page_icon="🎈",
)



with st.sidebar:
    st.title("Quel texte analyser ?")
    analyser_choice = st.radio("Quel texte analyser ?", ["Texte libre", "Avis dataset"])

    if analyser_choice == "Avis dataset":
        index_input = st.number_input("Numéro d'index", key="index_input", step=1, min_value=0, max_value=df.shape[0], on_change=index_input_callback)
        st.button("Aléatoire", on_click=aleatoire_callback)

    # if analyser_choice == "Texte libre":
    #     st.session_state['options'] = ""

review = st.text_area("Entrez un texte", height=50, max_chars=10000, key='options')
number = st.slider('Nombre de topics', step=1, min_value=1, max_value=15)

if review != "":
    detect_topic_btn = st.button("Détecter le sujet d'insatisfaction")
    if detect_topic_btn:
        st.title(number)
        st.info(review)
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", "70 °F", "1.2 °F")
        col2.metric("Wind", "9 mph", "-8%")
        col3.metric("Humidity", "86%", "4%")




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






# image = Image.open('image.png')
# st.image(image, caption='Sunrise by the mountains')


# if choice == "Upload":
#     st.title("Upload Your Dataset")
#     file = st.file_uploader("Upload Your Dataset")
#     if file:
#         df = pd.read_csv(file, index_col=None)
#         df.to_csv('dataset.csv', index=None)
#         st.dataframe(df)
#
# if choice == "Profiling":
#     st.title("Exploratory Data Analysis")
#     profile_df = df.profile_report()
#     st_profile_report(profile_df)
#
# if choice == "Modelling":
#     chosen_target = st.selectbox('Choose the Target Column', df.columns)
#     if st.button('Run Modelling'):
#         setup(df, target=chosen_target, silent=True)
#         setup_df = pull()
#         st.dataframe(setup_df)
#         best_model = compare_models()
#         compare_df = pull()
#         st.dataframe(compare_df)
#         save_model(best_model, 'best_model')
#
# if choice == "Download":
#     with open('best_model.pkl', 'rb') as f:
#         st.download_button('Download Model', f, file_name="best_model.pkl")



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