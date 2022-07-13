###############################################
# SENTIMENT ANALYSIS
###############################################

###################################
# Import Libraries
###################################

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud, ImageColorGenerator

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

###################################
# Reading Dataset
###################################

df = pd.read_csv("TEZ/clean_data")
df = df[["user_location","date","text"]]

df.isna().sum() # 613

# In the previous data, blanks represented an observation unit. Here, NaN returned a value, so it is removed from the data set.
df.dropna(inplace=True)

df.shape # (182546, 3)

##################################################
# Text Visualization
##################################################

drop_list = ["covid","vaccine","covidvaccine","amp","today","new","read","got","slot","vaccination","dose","covishield","people","vaccinated","availability"]
text_final = []
text = " ".join(i for i in df.text)
for i in text.split():
    if i not in drop_list: text_final.append(i)


def listToString(all_list):
    str1 = " "
    for words in all_list:
        str1 += words + " "
    return str1
text_final = listToString(text_final)

# *** all text wordcloud ***
mask = np.array(Image.open("TEZ/twitter/logoin.jpg").convert('RGBA'))
wordcloud = WordCloud(background_color="white", mode="RGB", max_words=1000, mask=mask).generate(text_final)
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[7, 7])
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("allsentimentin.png")

##################################################
# Sentiment Analysis
##################################################

# VADER (Valence Aware Dictionary and Sentiment Reasoner).

# nltk.download('vader_lexicon')
# labeled sentiment
# compound value between -1 and 1

sia = SentimentIntensityAnalyzer()
df["polarity_score"] = df["text"].apply(lambda x: sia.polarity_scores(x)["compound"])

df["sentiment_label"] = df["text"].apply(lambda x: "positive"  if sia.polarity_scores(x)["compound"] >= 0.05 else ("negative" if sia.polarity_scores(x)["compound"] <= -0.05 else "neutral"))

# most positive 5 tweets
df_sentiment = df.sort_values(by='polarity_score', ascending=False)[['text', 'sentiment_label', 'polarity_score']].reset_index(drop=True)
df_sentiment["text"].head()

# most negative 5 tweets
df_sentiment = df.sort_values(by='polarity_score', ascending=True)[['text', 'sentiment_label', 'polarity_score']].reset_index(drop=True)
df_sentiment["text"][4]

df_sentiment.sentiment_label.value_counts()

df.head()

def wordcloud_vis(dataframe_text, save_path, save = False):
    drop_list = ["covid", "vaccine", "covidvaccine", "amp", "today", "new", "read", "got", "slot", "vaccination",
                 "dose", "covishield", "people", "vaccinated", "availability"]
    text_final = []
    text = " ".join(i for i in dataframe_text)
    for i in text.split():
        if i not in drop_list: text_final.append(i)

    text_cleaned = listToString(text_final)

    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_cleaned)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    if save:
        wordcloud.to_file(save_path + ".png")


### wordcloud according to sentiment ###

# positive tweets word cloud
wordcloud_vis(df_sentiment[df_sentiment["sentiment_label"] == "positive"]["text"], "positive_tweets_word_cloud", save=True)

# negative tweets word cloud
wordcloud_vis(df_sentiment[df_sentiment["sentiment_label"] == "negative"]["text"], "negative_tweets_word_cloud", save=True)

# neutral tweets word cloud
wordcloud_vis(df_sentiment[df_sentiment["sentiment_label"] == "neutral"]["text"], "neutral_tweets_word_cloud", save=True)

### wordcloud for every country ###

df["user_location"].unique()
# ['United States', 'India', 'Canada', 'Australia', 'United Kingdom','China', 'Russia', 'Saudi Arabia',
# 'Turkey', 'Brazil', 'Indonesia', 'Netherlands', 'Germany', 'Italy', 'Switzerland', 'Spain', 'Mexico']

def wordcloud_flag(image_path,country_name,save_path,save=False):
    country_text = " ".join(review for review in df[df["user_location"]== country_name]["text"])

    drop_list = ["covid", "vaccine", "covidvaccine", "amp", "today", "new", "read", "got", "slot", "vaccination",
                 "dose", "covishield", "people", "vaccinated", "availability"]
    text_final = []

    for i in country_text.split():
        if i not in drop_list: text_final.append(i)

    text_cleaned = listToString(text_final)
    mask = np.array(Image.open(image_path).convert('RGBA'))
    wordcloud_usa = WordCloud(background_color="white", mode="RGBA", max_words=1000,mask=mask).generate(text_cleaned)
    image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[7, 7])
    plt.imshow(wordcloud_usa.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.show()
    if save:
        wordcloud_usa.to_file(save_path + ".png")


# United States
wordcloud_flag("country_flag/usa.jpg", "United States", "usa_flag_wordcloud", save=True)
# India
wordcloud_flag("country_flag/india.jpg", "India", "india_flag_wordcloud", save=True)
# Canada
wordcloud_flag("country_flag/canada.jpg", "Canada", "canada_flag_wordcloud", save=True)
# Australia
wordcloud_flag("country_flag/australia.jpg", "Australia", "australia_flag_wordcloud", save=True)
# United Kingdom
wordcloud_flag("country_flag/unitedkingdom.png", "United Kingdom", "unitedkingdom_flag_wordcloud", save=True)
# China
wordcloud_flag("country_flag/china.png", "China", "china_flag_wordcloud", save=True)
# Russia
wordcloud_flag("country_flag/russia.png", "Russia", "russia_flag_wordcloud", save=True)
# Saudi Arabia
wordcloud_flag("country_flag/saudiarabia.png", "Saudi Arabia", "saudiarabia_flag_wordcloud", save=True)
# Turkey
wordcloud_flag("country_flag/turkey.png", "Turkey", "turkey_flag_wordcloud", save=True)
# Brazil
wordcloud_flag("country_flag/brazil.png", "Brazil", "brazil_flag_wordcloud", save=True)
# Indonesia
wordcloud_flag("country_flag/indonesia.png", "Indonesia", "indonesia_flag_wordcloud", save=True)
# Netherlands
wordcloud_flag("country_flag/netherlands.jpg", "Netherlands", "netherlands_flag_wordcloud", save=True)
# Germany
wordcloud_flag("country_flag/germany.png", "Germany", "germany_flag_wordcloud", save=True)
# Italy
wordcloud_flag("country_flag/italy.png", "Italy", "italy_flag_wordcloud", save=True)
# Switzerland
wordcloud_flag("country_flag/switzerland.png", "Switzerland", "switzerland_flag_wordcloud", save=True)
# Spain
wordcloud_flag("country_flag/spain.jpg", "Spain", "spain_flag_wordcloud", save=True)
# Mexico
wordcloud_flag("country_flag/mexico.png", "Mexico", "mexico_flag_wordcloud", save=True)


df["sentiment_label"].value_counts()
'''              
positive    89394
neutral     58184
negative    34968
'''

# *** Sentiment Analysis of Tweets Sent Between 11 November 2020 - 1 January 2022 ***

plt.style.use('seaborn-darkgrid')
plt.title("Sentiment Analysis of Tweets Sent Between 1 November 2020 - 1 January 2022")
df["sentiment_label"].value_counts().plot(kind='barh', color = ['green','blue','red'])
plt.xlabel(r'$Sentiment $ $ Count$')
plt.ylabel(r'$Sentiment$')
plt.show()


# *** Sentiment Analysis of Daily Tweets ***

df_pos = df[df["sentiment_label"] == "positive"]
df_neg = df[df["sentiment_label"] == "negative"]
df_nötr = df[df["sentiment_label"] == "neutral"]

df_pos = df_pos.groupby("date").agg({"sentiment_label":"count"}).reset_index()
df_neg = df_neg.groupby("date").agg({"sentiment_label":"count"}).reset_index()
df_nötr = df_nötr.groupby("date").agg({"sentiment_label":"count"}).reset_index()

import matplotlib.dates as mdates
plt.figure(figsize = (19,9))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=8))
plt.plot(df_pos["date"],df_pos["sentiment_label"], label = "positive tweet count", linestyle = 'solid', color = 'green')
plt.plot(df_neg["date"],df_neg["sentiment_label"], label = "negative tweet count", linestyle = 'solid', color = 'red')
plt.plot(df_nötr["date"],df_nötr["sentiment_label"], label = "neutral tweet count", linestyle = 'solid', color = 'blue')
plt.xlabel(r'Date', fontsize = 16)
plt.ylabel(r'Tweet Count', fontsize = 16)
plt.legend(loc = 'upper right', fontsize=11)
plt.title("Sentiment Analysis of Daily Tweets", fontsize = 17)
plt.gcf().autofmt_xdate()
plt.show()

# *** Sentiment Analysis of Tweets by Country ***

country_sentiment = df.groupby(["user_location","sentiment_label"]).agg({"sentiment_label":"count"})
country_sentiment.rename(columns = {'sentiment_label':'count'}, inplace = True)
country_sentiment.reset_index(inplace = True)
countries = country_sentiment["user_location"].unique()

country_sentiment["normalized_count"] = (country_sentiment["count"]-country_sentiment["count"].min())/(country_sentiment["count"].max()-country_sentiment["count"].min())

fig = plt.figure(figsize=(22,9))
barWidth = 0.2
positive = country_sentiment[(country_sentiment["sentiment_label"] == "positive")]["normalized_count"]
negative = country_sentiment[(country_sentiment["sentiment_label"] == "negative")]["normalized_count"]
neutral = country_sentiment[(country_sentiment["sentiment_label"] == "neutral")]["normalized_count"]

# Set position of bar on X axis
pos1 = np.arange(len(positive))
pos2 = [x + barWidth for x in pos1]
pos3 = [x + barWidth for x in pos2]
# Make the plot
plt.bar(pos1, positive, color='green', width=barWidth, label='positive')
plt.bar(pos2, negative, color='red', width=barWidth, label='negative')
plt.bar(pos3, neutral, color='blue', width=barWidth, label='neutral')

plt.xlabel('Countries', fontweight='bold')
plt.ylabel('Sentiment Tweet Count', fontweight='bold')
plt.xticks([i+0.1 for i in range(len(neutral))],countries)
plt.ylim(0,1)
plt.title('Sentiment Analysis of Tweets by Country')
plt.legend()
plt.show()
fig.savefig('sentiment_count_countries.png')

df.to_csv("sentiment_label_data.csv")

########################################
# SENTIMENT COUNT BY COUNTRIES
########################################

countries = set(df.user_location.unique())
for i in countries:
    fig = plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-darkgrid')
    plt.title("i, fontsize = 30)
    df[(df["user_location"] == i)]["sentiment_label"].value_counts().sort_values().plot(kind='barh', color = ['blue','red','green'], fontsize = 19)
    plt.xlabel(r'$Sentiment $ $ Count$', fontsize = 21)
    plt.ylabel(r'$Sentiment$', fontsize = 21)
    plt.show()
    fig.savefig('sentiment_count_' + i + '.png')

