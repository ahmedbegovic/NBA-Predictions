import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Load the data for all games from the last 5 seasons
df = pd.read_csv('stats/team_games.csv')

#Choose the season (in this case it's going to be 2022-23 regular season)
#12022 - Pre season
#22022 - Regular season
#32022 - All-Star Weekend
#42022 - Playoffs
#52022 - Play-In tournament

df = df.loc[df['SEASON_ID'] == 22022]
#Drop the non-numeric and redundant features
df = df.drop(columns=['SEASON_ID','TEAM_ID','TEAM_NAME','GAME_ID','MATCHUP','WL','MIN','GAME_DATE'])

#NBA teams list
nba_teams = ['MIA','LAL','BOS','DEN','LAC','HOU','TOR','MIL','OKC','UTA','DAL','ORL','POR','IND','BKN','PHI','MEM','PHX','SAS','SAC','NOP','wAS','ATL','NYK','DET','CHA','CLE','CHI','MIN','GSW']

#Remove all games from data that include non-NBA teams, All-Star games
df = df[df['TEAM_ABBREVIATION'].str.contains("|".join(nba_teams))]

#Normalize the data
scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:], index=df.index)
normalized_df['Team'] = df['TEAM_ABBREVIATION']

#Weights for offensive composite score based on the statistics 
offensive_weights = {
    'PTS': 0.4, #Points
    'OREB': 0.1, #Offensive rebounds
    'AST': 0.2, #Assists
    'FG_PCT': 0.2, #Field goal percentage
    'FT_PCT': 0.1 #Free throw percentage
}

#Weights for defensive composite score based on the statistics 
defensive_weights = {
    'DREB': 0.4, #Defensive rebounds
    'STL': 0.2, #Steals
    'BLK': 0.2, #Blocks
    'TOV': 0.1, #Turnovers
    'PF': 0.1 #Personal fouls
}

# Calculate composite score

def offensive_composite_score():
    normalized_df['CompositeScore'] = (
        normalized_df['PTS'] * offensive_weights['PTS'] +
        normalized_df['OREB'] * offensive_weights['OREB'] +
        normalized_df['AST'] * offensive_weights['AST'] +
        normalized_df['FG_PCT'] * offensive_weights['FG_PCT'] +
        normalized_df['FT_PCT'] * offensive_weights['FT_PCT']
    )

def defensive_composite_score():
    normalized_df['CompositeScore'] = (
        normalized_df['DREB'] * defensive_weights['DREB'] +
        normalized_df['STL'] * defensive_weights['STL'] +
        normalized_df['BLK'] * defensive_weights['BLK'] +
        normalized_df['TOV'] * defensive_weights['TOV'] +
        normalized_df['PF'] * defensive_weights['PF']
    )

# Step 4: Visualize the team strengths
plt.figure(figsize=(14, 8))
#Calculate the composite score
#offensive_composite_score()
defensive_composite_score()
plt.bar(normalized_df['Team'], normalized_df['CompositeScore'], color='blue')
plt.xlabel('Team')
plt.ylabel('Defensive Composite Score')
plt.title('Team Strength Analysis for NBA teams in the season 2022-2023')
plt.xticks(rotation=90)
plt.show()