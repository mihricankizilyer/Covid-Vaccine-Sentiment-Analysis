import pandas as pd
import numpy as np

import warnings
import matplotlib.dates as mdates
import datetime as dt
from datetime import date

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows',150)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import matplotlib.pyplot as plt

df = pd.read_csv("TEZ/covid_death/death_rate.csv")

df.shape #  (191605, 67)

countries_gpd = ['India', 'United Kingdom', 'United States', 'Germany',
             'Canada', 'Netherlands', 'Turkey', 'China', 'Russia',
             'Brazil', 'Australia', 'Italy', 'Spain', 'Switzerland',
             'Saudi Arabia', 'Mexico', 'Indonesia']

df["country"] = df["location"].apply(lambda x: x if x in countries_gpd else "delete")

df = df[df["country"] != "delete"]

df = df.reset_index()

df = df[["country","date","new_cases","new_deaths"]]

df.shape # (14437, 4)

# *** DATASET ***
# start date: 2019-12-30
# end date:   2022-02-27

# *** ANALYSIS DATE ***
# "2020-11-01" - "2022-01-01"

df.shape # (5448, 17)

df = df[(df["date"] > "2020-11-01") & (df["date"] < "2022-01-01")]

df.isna().sum()
"""
country        0
date           0
new_cases      6
new_deaths    16
"""

df.dropna(inplace = True)

df.shape #  (7203, 4)

df.head(1)

df_deaths = df.groupby(["country","date"]).agg({"new_deaths":"sum"}).reset_index()
df_cases = df.groupby(["country","date"]).agg({"new_cases":"sum"}).reset_index()


# --- Number of Deaths by Country ---
fig = plt.figure(figsize = (18,9))
plt.barh("country", "new_deaths", data = df_deaths, color = "red")
plt.title("Number of Deaths by Country")
plt.xlabel("Deaths Count")
plt.ylabel("Countries")
plt.show()
fig.savefig('deaths_countries.png')

df_deaths = df_deaths.sort_values(by = "date")
df_deaths_groups = df_deaths.groupby("date").agg({"new_deaths":"sum"}).reset_index()

# --- Number of Deaths Due to Covid by Date ---
plt.style.use('seaborn-darkgrid')
fig = plt.figure(figsize = (15,6))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=12))
plt.plot(df_deaths_groups["date"],df_deaths_groups["new_deaths"], color = "red")
plt.title("Number of Deaths Due to Covid by Date")
plt.xlabel("Date")
plt.ylabel("Number of Deaths")
plt.gcf().autofmt_xdate()
plt.show()
fig.savefig('date_deaths.png')



# Data Source: https://ourworldindata.org/covid-deaths