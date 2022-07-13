###############################################
# EXPLORER DATA ANALYSIS
###############################################

###################################
# Import Libraries
###################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import date
from TEZ.countries import *
import re

from PIL import Image
import chart_studio.plotly
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud


# Visualization
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows',150)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###################################
# Reading Dataset
###################################

df_ = pd.read_csv("covidvaccinev51.csv", sep=',', usecols = ["user_location", "user_followers", "date", "text"])
df = df_.copy()

###################################
# Data Overview
###################################

def overview_df(dataframe, head = 5, tail = 5, observation = False):

    print("---------- SHAPE ----------") # (375651, 4)
    print(dataframe.shape)
    print("----------- NA -----------")
    print(dataframe.isnull().sum())
    print("---------- INFO ----------")
    print(dataframe.info())

    if observation:
        print("---------- HEAD ----------")
        print(dataframe.head(head))
        print("---------- TAIL ----------")
        print(dataframe.tail(tail))

overview_df(df)

###################################
# Data Cleaning
###################################

df = df.drop_duplicates(subset=['text'])

df.shape # (375262, 4)

# Values with missing observations are deleted
df.dropna(subset=['user_location','text','date'], inplace = True)

df.shape # (294560, 4)

df = df.reset_index(drop=True)

#####################
# Timeline
#####################

# If ‘coerce’, then invalid parsing will be set as NaN.
# neccessary number but have letter, there will be NaN.
df["date"] = pd.to_datetime(df['date'], errors='coerce')

df['date'] = df['date'].dt.strftime('%Y-%m-%d')

df.sort_values(by='date')

# Incorrect date entries are dropped because of nan value
df.dropna(subset=['date'], inplace = True)

# analysis date
df = df[(df["date"] > "2020-11-01") & (df["date"] < "2022-01-01")]

df = df.reset_index(drop = True)

df.shape # (265931, 4)

#####################
# Location
#####################

# In 2020, 17 of the top 20 countries were selected according to their GDP values.

fList = []

fList.append({'filter': INDIA_LIST, 'to': 'India' })

fList.append({'filter': UNITED_KINGDOM_LIST, 'to': 'United Kingdom' })

fList.append({'filter': UNITED_STATES_LIST, 'to': 'United States' })

fList.append({'filter': GERMANY_LIST, 'to': 'Germany' })

fList.append({'filter': CANADA_LIST, 'to': 'Canada' })

fList.append({'filter': NETHERLANDS_LIST, 'to': 'Netherlands' })

fList.append({'filter': TURKEY_LIST, 'to': 'Turkey' })

fList.append({'filter': CHINA_LIST, 'to': 'China' })

fList.append({'filter': RUSSIA_LIST, 'to': 'Russia' })

fList.append({'filter': BRAZIL_LIST, 'to': 'Brazil' })

fList.append({'filter':AUSTRALIA_LIST, 'to': 'Australia' })

fList.append({'filter':ITALY_LIST, 'to': 'Italy' })

fList.append({'filter':SPAIN_LIST, 'to': 'Spain' })

fList.append({'filter':SWITZERLAND_LIST, 'to': 'Switzerland' })

fList.append({'filter':SAUDI_ARABIA_LIST, 'to': 'Saudi Arabia' })

fList.append({'filter':MEXICO_LIST, 'to': 'Mexico' })

fList.append({'filter':INDONESIA_LIST, 'to': 'Indonesia' })

def filterLocation(loc, filterList):
    '''
    It takes the cities in the list one by one and returns the country in the list if it matches the given location value.
    If it does not match, it returns itself.

    Parameters
    ----------
    loc : string
        Indicates the city whose country is desired to be located.

    filterList : list
        It contains a dictionary containing the information of each country and city in the lists.
    Returns
    -------
    f['to']: str
        Returns what country the city is in
    loc: str
        Returns itself if no value is supplied

    '''
    for f in filterList:
        if any([(i in loc) for i in f['filter']]):
            return f['to']
    return loc

df['user_location'] = df['user_location'].apply(lambda x: filterLocation(x, fList))

countries = ['India', 'United Kingdom', 'United States', 'Germany',
             'Canada', 'Netherlands', 'Turkey', 'China', 'Russia',
             'Brazil', 'Australia', 'Italy', 'Spain', 'Switzerland',
             'Saudi Arabia', 'Mexico', 'Indonesia']

df["user_location"] = [country if country in countries else "DELETE" for country in df["user_location"]]

df.shape # (265931, 4)

df.drop(df[df["user_location"] == "DELETE"].index, inplace = True)

df.shape # (183159, 4)

df = df.reset_index(drop = True)

#####################
# Visualization
#####################

df_ = df.groupby("date").agg({"text":"count"})
df_ = df_.reset_index()

# --- Visualization of Daily Tweets ---

# with plot
plt.style.use('seaborn-darkgrid')
fig = plt.figure(figsize = (15,6))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=8))
plt.plot(df_["date"],df_["text"])
plt.title("Daily Tweet Count")
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.gcf().autofmt_xdate()
plt.show()
fig.savefig('VisualizationofDailyTweetsSonra.png')

# --- Number of Tweets by Country ---

df_country_tweet = df.groupby("user_location").agg({"text":"count"}).reset_index()
df_country_tweet = df_country_tweet.sort_values(by = "text", ascending = False).reset_index(drop=True)

tweet_count = df_country_tweet["text"]
country = df_country_tweet["user_location"]

fig = plt.figure(figsize = (12,5))
plt.barh("user_location", "text", data = df_country_tweet)
plt.title("Number of Tweets by Country")
plt.xlabel("Tweet Count")
plt.ylabel("Countries")
plt.show()
fig.savefig('tweet_count_countries.png')


###################################
# Text Processing
###################################

def text_processing(dataframe, col):

    dataframe[col] = dataframe[col].str.lower()  # normalization
    dataframe[col] = dataframe[col].str.replace('[^\w\s]', '', regex=True) # remove punctuation
    dataframe[col] = dataframe[col].str.replace('[^#\w\s]', '', regex=True) # remove hashtags
    dataframe[col] = dataframe[col].str.replace('\d','', regex = True) # remove numbers
    dataframe[col].apply(lambda x: TextBlob(x).words)  # tokenization

    # remove stopwords
    sw = stopwords.words('english')
    dataframe[col] = dataframe[col].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

    # lemmatization
    dataframe[col] = dataframe[col].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    # remove https
    dataframe[col] = dataframe[col].apply(lambda x: re.split('https[a-zA-Z]', str(x))[0])

    return dataframe[col]

df['text'] = text_processing(df, 'text')

remove_emoji = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          "]+", flags=re.UNICODE)

df['text'] = df['text'].apply(lambda x: remove_emoji.sub(r'', str(x))) # no emoji

df.to_csv("clean_data")