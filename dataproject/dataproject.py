import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import os 
from dstapi import DstApi
import seaborn as sns

filename = 'data/sport.csv'

# We tell Python that ';' is meant to be a seperator
sport = pd.read_csv(filename, sep=';')

# We drop the first 3 rows
sport = sport.drop([0,1,2])

# We want to drop the first two columns
sport_drop = ['.1', '.2']

# We remove white spaces 
sport.columns = sport.columns.str.strip()

# We drop the columns 
sport = sport.drop(columns=sport_drop)

# Renaming the first column
sport.rename(columns = {'.3':'event'}, inplace=True)

# Reset the index
sport.reset_index(inplace=True, drop=True)

# Having variables labeled as numbers is problematic. 
# Thus, we rename our year-variables. 
# To do that we use a dictonary.
col_dict = {}
for i in range(2014,2022+1):
    col_dict[str(i)] = f'sport_{i}'
col_dict

# Renaming our year-variables based on our dictionary
sport.rename(columns=col_dict,inplace=True)

# The data is better in a long format since we have two identifiers, event and year

# Arranging the data to a long dataset
sport_long = pd.wide_to_long(sport, stubnames='sport_', i='event', j='year')

# FETCHING

# Fetching the data
ind = DstApi('IDRTIL01')

params = ind._define_base_params(language='en')

# Fetching the appropiate data
params = {'table': 'idrtil01',
 'format': 'BULK',
 'lang': 'en',
 'variables': [{'code': 'SPORTS', 'values': ['SPO005', 'SPO020', 'SPO025', 'SPO035', 'SPO050','SPO065','SPO070','SPO090']},
  {'code': 'TILSKUER', 'values': ['ENH15']},
  {'code': 'Tid', 'values': ['*']}]}
sport_api = ind.get_data(params=params)

# We rename the variables and drop irrelevant one
sport_api.rename(columns={'INDHOLD':'avr_attend'}, inplace=True)
sport_api.rename(columns={'TID':'year'}, inplace=True)
sport_api.rename(columns={'SPORTS':'event'}, inplace=True)
sport_api.drop(columns=['TILSKUER'], inplace=True)

# Changing the names of the events
new_names = {
    'Ice hockey - The Ice Hockey League - Season total - men':'icehockey_league',
    'Ice hockey - International (In Denmark) - men':'icehockey_national',
    'Basketball - International (In Denmark) - men':'basketball_national',
    'Football - International (In Denmark) - men':'football_national',
    'Football - Superleague - men':'football_league',
    'Basketball -The Basketball League season total - men':'basketball_league',
    'Handball - International (In Denmark) - men':'handball_national',
    'Handball - The Handball League season total - men':'handball_league'
}
# Applying the new names
sport_api['event'] = sport_api['event'].replace(new_names)

# We establish a loop to convert seasonal accounting into annual 
new_year_loop = {}
for year in range(2006,2023+1):
    year_str = str(year)
    next_year_str = str(year+1)
    new_year_loop[f'{year_str}/{next_year_str}'] = year_str
sport_api_annual = sport_api.copy()
sport_api_annual['year'] = sport_api_annual['year'].replace(new_year_loop)    

# we drop the non-numeric attend-variable
sport_api_annual = sport_api_annual.drop(columns=['avr_attend'])

# We group the data by 'year' and 'event'
grouped_data = sport_api_annual.groupby(['year', 'event'])['avr_attend_numeric'].sum().reset_index()

# Select and filter only the 'national' and 'league' observations for each sport, then create a new variable for each
sports = ['icehockey', 'football', 'basketball', 'handball']
for sport in sports:
    filtered_data = grouped_data[grouped_data['event'].str.contains(sport)]
    overall_attend = filtered_data.groupby('year')['avr_attend_numeric'].sum().reset_index()
    overall_attend.rename(columns={'avr_attend_numeric': f'{sport}_overall_attend'}, inplace=True)
    attend_sum = sport_api_annual.merge(overall_attend, on='year', how='left')
