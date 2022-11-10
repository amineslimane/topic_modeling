import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text
import pickle

DATASET_FILE = "dataset_cleaned.csv"
df = pd.read_csv(DATASET_FILE)
df = df[df.stars < 3]
vectorizer = TfidfVectorizer(ngram_range = (1,1), max_df = .8, min_df = .02)
data = vectorizer.fit_transform(df.text_cleaned)
matrix_df = pd.DataFrame(data.toarray(), columns=vectorizer.get_feature_names())
matrix_df.index = df.index
def display_topics(model, feature_names, num_top_words,topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i] \
             for i in topic.argsort()[:-num_top_words - 1:-1]]))
topics =  ['Staff management', 'Food Quality', 'Pizza', 'Menu Chicken', 'Quality', 'Service time',
          'Burger', 'Waiting Time', 'Experience', 'Drinks', 'Ordering & Delivery to table', 'Location',
          'Customer Service',  'Sushi and Rice', 'Place Environnement']

nmf_model = NMF(15)
doc_topic = nmf_model.fit_transform(matrix_df)
display_topics(nmf_model, vectorizer.get_feature_names(), 10, topics)
with open('model_yasmine','wb') as file:
    pickle.dump(nmf_model, file)
with open('vectorizer_yasmine','wb') as file:
    pickle.dump(vectorizer, file)