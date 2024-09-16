# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:47:14 2021

@author: CS
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#from datetime import datetime
#import re
import plotly.express as px
import plotly.graph_objects as go


df =pd.read_csv('SNGPL Meter Reading with correct Date and Time format.csv')
print(df.head(20))
# set dat as Index
df.set_index('Date_and_Time', inplace = True)

# summarize
print(df.shape)


df['Interp_reading']  = df['Meter_Reading'].interpolate()  
df['Interp_reading'] = df['Interp_reading'].apply(np.ceil)
# print("----------------------------------------------------------")
# print("After interpolate the meter reading data, dataFrame lool like")
# print(df.head(20))

#Estimating the gas in m3
df['M3_reading']=df['Interp_reading']/1000
# print("----------------------------------------------------------")
# print("After adding m3 meter reading data, dataFrame lool like")
# print(df.head(20))

#Estimating the meter reading Difference
df['M3_difference'] = df['M3_reading'].diff()
# print("----------------------------------------------------------")
# print("After taking difference of meter reading, dataFrame lool like")
# print(df.head(20))

#Estimating the gas in hm3
df['Hm3_reading']=df['M3_difference']/100
# print("----------------------------------------------------------")
# print("After estimating hm3 of meter reading, dataFrame lool like")
# print(df.head(20))

#estimating the gas in MMBTU
GCV=990
df['Mmbtu_reading']=df['Hm3_reading']*GCV/281.7385
# print("----------------------------------------------------------")
# print("After estimating mmbtu of meter reading, dataFrame lool like")
# print(df.head(20))

df['MegaJ_reading']=df['Mmbtu_reading']*1055.055853
df['Kwh_reading']=df['Mmbtu_reading']*293.07

print("----------------------------------------------------------")
print("After estimating mega joules and kwh of meter reading, dataFrame lool like")
print(df.head(20))
# save updated dataset
df.to_csv('SNGPL Meter Reading with all Columns.csv')




