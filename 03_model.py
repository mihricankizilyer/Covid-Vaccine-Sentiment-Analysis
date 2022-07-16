##################################################
# Sentiment Modeling
##################################################

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer

from textblob import Word, TextBlob
from wordcloud import WordCloud

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, \
    plot_roc_curve, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

###############################
# Reading Data
###############################

df_ = pd.read_csv("TEZ/sentiment_label_data.csv")

df_.dropna(inplace = True)

df = df_.copy()

df = df[["user_location","date","text","polarity_score","sentiment_label"]]

###############################
# Feature Engineering
###############################

### TF-IDF ###
vectorizer = TfidfVectorizer(max_features=2000, max_df = 0.90, min_df = 10, stop_words=stopwords.words('english'))
features = vectorizer.fit_transform(df['text'].values).toarray()

# Categoric label convert to numeric value
df["sentiment_label_encoding"] = LabelEncoder().fit_transform(df["sentiment_label"])

# 0: negative
# 1: neutral
# 2: positive

X = features # independent variable
y = df["sentiment_label"] # dependent variable

# The model is installed on the train set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, df['sentiment_label'].values, test_size=0.20, random_state=42)
# Due to the large number of data sets, the CV method trains the model for a long time. For this reason, modeling was done with the hold out method.

###############################
# MODELLING
###############################

###############################
# Multi Logistic Regression
###############################

model_multilog_reg = LogisticRegression(random_state=1, multi_class='multinomial', solver='newton-cg').fit(X_train, y_train)

# --- Success Evaluation ---

y_pred = model_multilog_reg.predict(X_train)

print('Train Accuracy Score:', metrics.accuracy_score(y_train, y_pred))
# Train Accuracy Score: 0.8723932472691162

# test
y_prob = model_multilog_reg.predict_proba(X_test)[:,1]

# y_pred for other metrics
y_pred = model_multilog_reg.predict(X_test)

# Confusion Matrix
def plot_confision_matrix(y, y_red):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy score: {0}".format((acc), size=10))
    plt.show()

plot_confision_matrix(y_test, y_pred)

print('Test Accuracy Score:', metrics.accuracy_score(y_test, y_pred))
# Test Accuracy Score: 0.8164539007092199

print('Precision Score:', metrics.precision_score(y_test, y_pred, average='weighted'))
# average parameter neccessary for multi class label
# Precision Score: 0.8194892160062097

print('Recall Score:', metrics.recall_score(y_test, y_pred, average='weighted'))
# Recall Score: 0.8164539007092199

print('F1-Score Score:', metrics.f1_score(y_test, y_pred, average='weighted'))
# F1-Score Score: 0.8103142052806553

classification_report(y_test, y_pred)

###############################
# Random Forest
###############################

# RF Trial:
rf = RandomForestClassifier(criterion = 'entropy', random_state = 1)

rf_model = rf.fit(X_train,y_train)

# test accuracy
y_pred = rf_model.predict(X_test)

print('Test Accuracy Score:', metrics.accuracy_score(y_test, y_pred))
# Test Accuracy Score: 0.8039716312056737

print('Precision Score:', metrics.precision_score(y_test, y_pred, average='weighted'))
# Precision Score: 0.8237286950147258

print('Recall Score:', metrics.recall_score(y_test, y_pred, average='weighted'))
# Recall Score: 0.8039716312056737

print('F1-Score Score:', metrics.f1_score(y_test, y_pred, average='weighted'))
# F1-Score Score: 0.7959791231283259

# Confusion Matrix
def plot_confision_matrix(y, y_red):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy score: {0}".format((acc), size=10))
    plt.show()

plot_confision_matrix(y_test, y_pred)

classification_report(y_test, y_pred)

###############################
# Naive Bayes
###############################

clf = MultinomialNB().fit(X_train, y_train)

y_pred = clf.predict(X_train)

print('Train Accuracy Score:', metrics.accuracy_score(y_train, y_pred))
# Train Accuracy Score: 0.7828060717832317

# test
y_prob = clf.predict_proba(X_test)[:,1]

# y_pred for other metrics
y_pred = clf.predict(X_test)

# Confusion Matrix
def plot_confision_matrix(y, y_red):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy score: {0}".format((acc), size=10))
    plt.show()

plot_confision_matrix(y_test, y_pred)

print('Test Accuracy Score:', metrics.accuracy_score(y_test, y_pred))
# Test Accuracy Score: 0.7299290780141844

print('Precision Score:', metrics.precision_score(y_test, y_pred, average='weighted'))
# average parameter neccessary for multi class label
# Precision Score: 0.749021906040851

print('Recall Score:', metrics.recall_score(y_test, y_pred, average='weighted'))
# Recall Score: 0.7299290780141844

print('F1-Score Score:', metrics.f1_score(y_test, y_pred, average='weighted'))
# F1-Score Score: 0.7124092100637126

classification_report(y_test, y_pred)

###############################
# Ridge
###############################

clf = RidgeClassifier().fit(X_train, y_train)

y_pred = clf.predict(X_train)

print('Train Accuracy Score:', metrics.accuracy_score(y_train, y_pred))
# Train Accuracy Score: 0.8784934033196198

# y_pred for other metrics
y_pred = clf.predict(X_test)

# Confusion Matrix
def plot_confision_matrix(y, y_red):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy score: {0}".format((acc), size=10))
    plt.show()

plot_confision_matrix(y_test, y_pred)

print('Test Accuracy Score:', metrics.accuracy_score(y_test, y_pred))
# Test Accuracy Score: 0.8212765957446808

print('Precision Score:', metrics.precision_score(y_test, y_pred, average='weighted'))
# average parameter neccessary for multi class label
# Precision Score: 0.8290152517254779

print('Recall Score:', metrics.recall_score(y_test, y_pred, average='weighted'))
# Recall Score: 0.8212765957446808

print('F1-Score Score:', metrics.f1_score(y_test, y_pred, average='weighted'))
# F1-Score Score: 0.8170223119267128

classification_report(y_test, y_pred)