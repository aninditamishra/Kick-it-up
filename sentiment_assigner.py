# CSCE 676 Fall 2019 Final Project
# sentiment_assigner.py

# This file is the custom heuristic outlined in our final report. It is designed to assign a sentiment of positive,
# neutral, or negative depending on whether a project was successful, failed, or the percentage of the goal amount that
# the project was able to raise.

import pandas as pd

def isneutral(pledged, goal):
    tempvalue = pledged / goal
    if tempvalue >= 0.3:
        return True
    else:
        return False


blurb_sentiment_list = []
name_sentiment_list = []

kickstarterdata = pd.read_csv('kickstarter.csv')
kickstarterdf = pd.DataFrame(kickstarterdata)

goaldf = kickstarterdf[['goal']]
pledged_amountdf = kickstarterdf[['usd_pledged']]
statedf = kickstarterdf[['state']]

for index, row in statedf.iterrows():
    statestring = str(row['state'])
    if statestring == 'failed':
        goalvalue = float(goaldf.iloc[index])
        pledgedvalue = float(pledged_amountdf.iloc[index])
        if isneutral(pledgedvalue, goalvalue):
            blurb_sentiment_list.append('neutral')
            name_sentiment_list.append('neutral')
        else:
            blurb_sentiment_list.append('negative')
            name_sentiment_list.append('negative')

    else:
        blurb_sentiment_list.append('positive')
        name_sentiment_list.append('positive')

kickstarterdf['blurb_sentiment'] = blurb_sentiment_list
kickstarterdf['name_sentiment'] = name_sentiment_list

kickstarterdf.to_csv(r'kickstarter_sentiment.csv', index=False)

positivecounter = 0
neutralcounter = 0
negativecounter = 0

for item in name_sentiment_list:
    if item == 'positive':
        positivecounter += 1
    elif item == 'neutral':
        neutralcounter += 1
    elif item == 'negative':
        negativecounter += 1

print("Number of Positives: " + str(positivecounter))
print("Number of Neutrals: " + str(neutralcounter))
print("Number of Negatives: " + str(negativecounter))

