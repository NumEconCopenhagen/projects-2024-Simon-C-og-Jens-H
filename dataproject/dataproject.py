filename = 'data/sport.csv'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

import ipywidgets as widgets
import os 
from dstapi import DstApi
import seaborn as sns

def plot_regression(df, sport):
    plt.figure(figsize=(10, 6))
    sns.regplot(x='sport_', y=sport+'_overall_attend', data=df[df['event'] == sport], scatter_kws={'alpha': 0.9})
    plt.xlabel('Memberships')
    plt.ylabel('Average Attendance')
    plt.title(sport.capitalize())
    plt.show()

