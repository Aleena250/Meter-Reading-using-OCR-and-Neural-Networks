# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:52:15 2021

@author: EIG
"""
import pandas as pd
from pandas import Grouper
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import plotly.express as px
import plotly.graph_objects as go

df =pd.read_csv('SNGPL Meter Reading with all Columns.csv',parse_dates=['Date_and_Time'],dayfirst=True)
print(df.head(20))
# set dat as Index
df.set_index('Date_and_Time', inplace = True)
print(df.head(20))



selected_columns = df["M3_difference"]
df_copy = selected_columns.copy()
print(df_copy.head(20))
df_copy= df_copy.reset_index(drop=False)
print(df_copy.head())


# calculate Hourly  gas usage in m3

df_copy_hour=df_copy.copy()
df_copy_hour['Date_and_Time'] = pd.to_datetime(df_copy_hour['Date_and_Time']).dt.hour
df_copy_hour.rename(columns={"Date_and_Time": "hours"},inplace=True)
print(df_copy_hour)
df_copy_hour=df_copy_hour.groupby(df_copy_hour['hours'],sort=False).agg(
    # Get sum of the interpolate meter reading in m3 column for each group
    sum_consumption_perHour=('M3_difference', sum))
df_copy_hour=df_copy_hour.reset_index(drop=False)
print(df_copy_hour)


#calculate daily gas consumption in m3
df_copy_day=df_copy.copy()
df_copy_day['Date_and_Time'] = pd.to_datetime(df_copy_day['Date_and_Time']).dt.date
df_copy_day.rename(columns={"Date_and_Time": "days"},inplace=True)
print(df_copy_day)
df_copy_day=df_copy_day.groupby('days',sort=False).agg(
    # Get sum of the interpolate meter reading in m3 column for each group
    sum_consumption_perDay=('M3_difference', sum))
df_copy_day=df_copy_day.reset_index(drop=False)
print(df_copy_day)


# calculate monthly gas usage in m3

df_copy_month=df_copy.copy()
df_copy_month['Date_and_Time'] = pd.to_datetime(df_copy_month['Date_and_Time']).dt.month
df_copy_month.rename(columns={"Date_and_Time": "months"},inplace=True)
print(df_copy_month)
df_copy_month=df_copy_month.groupby(df_copy_month['months'],sort=False).agg(
    # Get sum of the interpolate meter reading in m3 column for each group
    sum_consumption_perMonth=('M3_difference', sum))
df_copy_month=df_copy_month.reset_index(drop=False)
df_copy_month["hm3"]=df_copy_month["sum_consumption_perMonth"]/100
print(df_copy_month)



