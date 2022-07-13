import numpy as np
import pandas as pd
import warnings
import matplotlib.dates as mdates
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows',150)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import matplotlib.pyplot as plt

df = pd.read_csv("TEZ/covid_test/tests_dataset.csv")

df = df[['location', 'date','new_tests']]

df.head()

df.shape # (191834, 9)

countries_gpd = ['India', 'United Kingdom', 'United States', 'Germany',
             'Canada', 'Netherlands', 'Turkey', 'China', 'Russia',
             'Brazil', 'Australia', 'Italy', 'Spain', 'Switzerland',
             'Saudi Arabia', 'Mexico', 'Indonesia']

df["location"] = df["location"].apply(lambda x: x if x in countries_gpd else "delete")

df = df[df["location"] != "delete"]

df = df.reset_index()

# *** DATASET ***
# start date: 2019-12-30
# end date:   2022-02-27

# *** ANALYSIS DATE ***
# "2020-11-01" - "2022-01-01"

df = df[(df["date"] > "2020-11-01") & (df["date"] < "2022-01-01")]

df.isna().sum() # 0
"""
location        0
date            0
new_tests    1388
"""

df.dropna(inplace = True)

df.shape # (5837, 4)

df.head()

df_ = df.groupby(["location","date"]).agg({"new_tests":"sum"}).reset_index()


# --- Number of Test Count by Country ---
fig = plt.figure(figsize = (15,7))
plt.barh("location", "new_tests", data = df_, color = "purple")
plt.title("Number of Test by Country")
plt.xlabel("Test Count")
plt.ylabel("Countries")
plt.show()
fig.savefig('test_countries.png')


df = df.sort_values(by = "date")
df_groups = df.groupby("date").agg({"new_tests":"sum"}).reset_index()

# --- Number of Deaths Due to Covid by Date ---
plt.style.use('seaborn-darkgrid')
fig = plt.figure(figsize = (15,6))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=12))
plt.plot(df_groups["date"],df_groups["new_tests"], color = "purple")
plt.title("Number of Test by Date")
plt.xlabel("Date")
plt.ylabel("Number of Test")
plt.gcf().autofmt_xdate()
plt.show()
fig.savefig('date_test.png')




