# CSCE 676 Fall 2019 Final Project
# sentiment_predictor.py

# This file is designed to load previously trained random forest ensemble models and predict sentiment labels of
# positive, neutral, or negative for all name and blurb strings in the test set of the Kickstarter data

import re
import pandas as pd
# import nltk
# nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


clean_blurb_features = []
clean_name_features = []

kickstarterdata = pd.read_csv('kickstarter_sentiment.csv')
kickstarterdf = pd.DataFrame(kickstarterdata)

blurbdf = kickstarterdf[['blurb']]
namedf = kickstarterdf[['name']]
blurb_sentimentdf = kickstarterdf[['blurb_sentiment']]
name_sentimentdf = kickstarterdf[['name_sentiment']]

blurb_features = blurbdf.values
name_features = namedf.values
blurb_sentiment_features = blurb_sentimentdf.values
name_sentiment_features = name_sentimentdf.values

# Regex responsible for pre-processing blurb string data to remove extraneous information
for sentence in range(0, len(blurb_features)):
    processed_feature = re.sub(r'\W', ' ', str(blurb_features[sentence]))
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    processed_feature = processed_feature.lower()
    clean_blurb_features.append(processed_feature)

# Regex responsible for pre-processing name string data to remove extraneous information
for name in range(0, len(name_features)):
    processed_feature = re.sub(r'\W', ' ', str(name_features[name]))
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    processed_feature = processed_feature.lower()
    clean_name_features.append(processed_feature)

# print("Completed cleaning data.")

# print("Beginning vectorizing data.")
blurb_vectorizer = TfidfVectorizer(max_features=20000, min_df=5, max_df=0.5, stop_words=stopwords.words('english'))
name_vectorizer = TfidfVectorizer(max_features=20000, min_df=5, max_df=0.5, stop_words=stopwords.words('english'))

clean_blurb_features = blurb_vectorizer.fit_transform(clean_blurb_features).toarray()
clean_name_features = name_vectorizer.fit_transform(clean_name_features).toarray()
# print("Completed vectorizing data.")

# print("Clean Name Features: ")
# print(str(len(clean_name_features)))
# print(clean_name_features)
# print()
# print("Clean Blurb Features: ")
# print(str(len(clean_blurb_features)))
# print(clean_blurb_features)

print("Beginning loading models.")
name_loaded_model = pickle.load(open("name_randomforest_model.sav", 'rb'))
blurb_loaded_model = pickle.load(open("blurb_randomforest_model.sav", 'rb'))
print("Completed loading models.")

print("Beginning predictions.")
name_predictions = name_loaded_model.predict(clean_name_features)
blurb_predictions = blurb_loaded_model.predict(clean_blurb_features)
print("Completed predictions")

print("Name Predictions: ")
print(len(name_predictions))
print(name_predictions)
print("Blurb Predictions: ")
print(len(blurb_predictions))
print(blurb_predictions)

kickstarterdf['name_sentiment_predictions'] = name_predictions
kickstarterdf['blurb_sentiment_predictions'] = blurb_predictions
kickstarterdf.to_csv(r'kickstarter_final1.csv', index=False)

print("Name Confusion Matrix: ")
print(confusion_matrix(name_sentiment_features, name_predictions))
print("Name Classification Report: ")
print(classification_report(name_sentiment_features, name_predictions))
print("Name Accuracy Score: ")
print(accuracy_score(name_sentiment_features, name_predictions))

print("Blurb Confusion Matrix: ")
print(confusion_matrix(blurb_sentiment_features, blurb_predictions))
print("Blurb Classification Report: ")
print(classification_report(blurb_sentiment_features, blurb_predictions))
print("Blurb Accuracy Score: ")
print(accuracy_score(blurb_sentiment_features, blurb_predictions))
