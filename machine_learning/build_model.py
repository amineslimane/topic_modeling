from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.decomposition import NMF
import pickle


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


# Construction du modèle
def build_model(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=.8, min_df=.02)
    data = vectorizer.fit_transform(df.text_cleaned)
    matrix_df = pd.DataFrame(data.toarray(), columns=vectorizer.get_feature_names())
    matrix_df.index = df.index

    nmf_model = NMF(15, max_iter=600)
    doc_topic = nmf_model.fit_transform(matrix_df)
    display_predicted_topics(nmf_model, vectorizer.get_feature_names(), 10, topics)
    with open('../nmf_model/model','wb') as file:
        pickle.dump(nmf_model, file)
    with open('../nmf_model/vectorizer','wb') as file:
        pickle.dump(vectorizer, file)


def display_predicted_topics(model, feature_names, num_top_words,topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i] \
             for i in topic.argsort()[:-num_top_words - 1:-1]]))


# Construction du modèle
dataset_negative_df = pd.read_csv("../data/dataset_negative.csv")
build_model(dataset_negative_df)