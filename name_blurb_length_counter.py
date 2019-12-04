# CSCE 676 Fall 2019 Final Project
# name_blurb_length_counter.py

# This file is designed to work with a .csv file of cleaned data, identify the "name" and "blurb" columns within
# the .csv, count the number of characters within each of the respective pieces of data, then append a new column
# to the .csv file with the respective counts for each column then export that new .csv

import pandas as pd

def charactercounter(string):
    return(float(len(string)))


blurb_length = []
name_length = []

kickstarterdata = pd.read_csv('cleanedCSV.csv')
kickstarterdf = pd.DataFrame(kickstarterdata)

blurbdf = kickstarterdf[['blurb']]
namedf = kickstarterdf[['name']]

for index, row in blurbdf.iterrows():
    blurbstring = str(row['blurb'])
    blurbvalue = charactercounter(blurbstring)
    blurb_length.append(blurbvalue)

for index, row in namedf.iterrows():
    namestring = str(row['name'])
    namevalue = charactercounter(namestring)
    name_length.append(namevalue)

kickstarterdf['blurb_length'] = blurb_length
kickstarterdf['name_length'] = name_length

kickstarterdf.to_csv(r'kickstarter.csv', index=False)





