#Code to fetch data for models learning

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import LeagueID
from nba_api.stats.static import players
import pandas as pd
import time

# 1. Performance prediction - LeBron James, fetching his stats for each game from 2019 to 2024
def lebron_stats():
    seasons = ['2007-08','2008-09','2009-10','2010-11','2011-12','2012-13','2013-14','2014-15','2015-16','2016-17','2017-18','2018-19','2019-20','2020-21','2021-22','2022-23','2023-24']
    total_stats = pd.DataFrame()

# Get the data for all 5 seasons
    for season in seasons:
        player_log = playergamelog.PlayerGameLog(2544, season)
        lebron_season_log = player_log.get_data_frames()[0]
        total_stats = pd.concat([total_stats,lebron_season_log], ignore_index= True)

    total_stats.to_csv('lebron_16_seasons.csv', index= False)

# 2. Team win predictions

def team_wins():

    #Get game results data for every game from 2019 to 2024
    total_games = pd.DataFrame()
    seasons = ['2019-20','2020-21','2021-22','2022-23','2023-24']
    for season in seasons:
        gamefinder = leaguegamefinder.LeagueGameFinder(league_id_nullable=LeagueID.nba, season_nullable = season)
        games = gamefinder.get_data_frames()[0]
        total_games = pd.concat([total_games, games], ignore_index = True)

    total_games.to_csv('team_games.csv', index= False)

# 3. Player similarity prediction

def similiar_players():

    all_players = pd.DataFrame()
    current_season = '2023-24'

    #Get all current active players
    active_players = players.get_active_players()

    for player in active_players:
        
        pid = player["id"]
        stats = playergamelog.PlayerGameLog(player_id= pid, season = current_season)
        stats_data = stats.get_data_frames()[0]

        #Find the average stats of the player during the current season
        current_season_stats = stats_data.drop(columns=['Game_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'VIDEO_AVAILABLE']).mean(numeric_only=True)

        #Data manipulation
        current_season_stats = current_season_stats.round(2)
        current_season_stats = pd.DataFrame(current_season_stats).transpose()
        #Save to file
        all_players = pd.concat([all_players,current_season_stats], ignore_index = True)
        all_players.to_csv("player_stats.csv", index = False)
        #To stop being timed out by the stats.nba.com, timer added to API calls
        #time.sleep(8)

# 4. Team strength


# Run the functions to get datasets
lebron_stats()
#team_wins()
#similiar_players()