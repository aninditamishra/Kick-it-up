# CSCE 676 Fall 2019 Final Project
# sentiment_trainer.py

# This file is designed to train the random forest ensemble classifiers for both name sentiment and blurb sentiment
# respectively and output the models as .sav files. Note that it was necessary to separate the training and prediction
# phases of computation because the average computer would run out of memory and error out of attempting to complete
# both of these steps consecutively.

import re
import pandas as pd
# import nltk
# nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

clean_blurb_features = []
clean_name_features = []

kickstarterdata = pd.read_csv('kickstarter_sentiment.csv')
kickstarterdf = pd.DataFrame(kickstarterdata)

blurbdf = kickstarterdf[['blurb']]
namedf = kickstarterdf[['name']]
labeldf = kickstarterdf[['state']]

blurb_sentimentdf = kickstarterdf[['blurb_sentiment']]
name_sentimentdf = kickstarterdf[['name_sentiment']]

blurb_features = blurbdf.values
name_features = namedf.values
label_features = labeldf.values

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

# print("Beginning splitting data.")
X_train_name, X_test_name, y_train_name, y_test_name = train_test_split(clean_name_features, name_sentiment_features,
                                                                        test_size=0.9, random_state=0)
X_train_blurb, X_test_blurb, y_train_blurb, y_test_blurb = train_test_split(clean_blurb_features,
                                                                            blurb_sentiment_features, test_size=0.9,
                                                                            random_state=0)
# print("Completed splitting data.")

name_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
blurb_classifier = RandomForestClassifier(n_estimators=200, random_state=0)

# print("Beginning training models.")
name_classifier.fit(X_train_name, y_train_name.ravel())
blurb_classifier.fit(X_train_blurb, y_train_blurb.ravel())
# print("Completed training models.")

namemodelfile = "name_randomforest_model.sav"
pickle.dump(name_classifier, open(namemodelfile, 'wb'))

blurbmodelfile = "blurb_randomforest_model.sav"
pickle.dump(blurb_classifier, open(blurbmodelfile, 'wb'))


