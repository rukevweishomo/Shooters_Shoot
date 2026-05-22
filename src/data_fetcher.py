from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import players
from nba_api.stats.static import teams
import pandas as pd
import time
import traceback

def get_team_id(team_name):
    teams_list = teams.get_teams()
    if team_name.lower() == 'all':
        return 0
    for team in teams_list:
        if team['full_name'] == team_name:
            return team['id']
    raise ValueError(f"Team '{team_name}' not found.")

def get_player_id(player_name):
    if player_name.lower() == 'all':
        return 0
    matching_players = players.find_players_by_full_name(player_name)
    if not matching_players:
        raise ValueError(f"No player found with name {player_name}")
    elif len(matching_players) > 1:
        print(f"Multiple players found with name '{player_name}':")
        for idx, player in enumerate(matching_players, start=1):
            print(f"{idx}: {player['full_name']} (ID: {player['id']})")
        selected_idx = int(input("Enter the number corresponding to the correct player: "))
        return matching_players[selected_idx - 1]['id']
    else:
        return matching_players[0]['id']
    
def format_season(season):
    season = str(season)
    if len(season) == 4 and season.isdigit():
        return f"{season}-{str(int(season) + 1)[-2:]}"
    elif len(season) == 7 and season[4] == '-' and season[:4].isdigit() and season[5:].isdigit():
        return season
    else:
        raise ValueError("Season must be in 'YYYY' or 'YYYY-YY' format.")
    
    
def fetch_shot_data(player_id, team_id, season, season_type, context_measure_simple) -> pd.DataFrame:
    shot_data = shotchartdetail.ShotChartDetail(
        team_id=team_id,
        player_id=player_id,
        season_nullable=season,
        season_type_all_star=season_type,
        context_measure_simple=context_measure_simple
    )
    return shot_data.get_data_frames()[0]

def fetch_multiple_seasons(player_id, team_id, start_year, end_year, season_type, context_measure_simple) -> pd.DataFrame:
    all_data = []

    for year in range(start_year, end_year + 1):
        season = format_season(year)
        print(f"Fetching data for {season} season...")
        try:
            data = fetch_shot_data(player_id, team_id, season, season_type, context_measure_simple)
            data['SEASON'] = season
            all_data.append(data)
        except Exception as e:
            print(f"Error fetching data for {season}: {e}")
            traceback.print_exc()
        time.sleep(5)

    if not all_data:
        print("No data fetched.")
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)