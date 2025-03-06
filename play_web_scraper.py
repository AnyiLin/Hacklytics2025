import requests
import json

def get_game_id(date, teams):
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={date}"
    headers = {'User-Agent': 'Mozilla/5.0'}  # using a common user agent header

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    for event in data.get('events', []):
        teams_info = event.get('competitions', [{}])[0].get('competitors', [])
        home_team = teams_info[0].get('team', {}).get('abbreviation')
        away_team = teams_info[1].get('team', {}).get('abbreviation')
        if home_team in teams or away_team in teams:
            if (event.get("competitions")[0].get("playByPlayAvailable")):
                return event.get("id")
            else:
                return None
    
    return None

def find_clock_elapsed_time(start, end):
    start_time = start.split(":")
    end_time = end.split(":")
    start_time = [int(i) for i in start_time]
    end_time = [int(i) for i in end_time]

    elapsed_time = (start_time[0] - end_time[0]) * 60 + (start_time[1] - end_time[1])
    return elapsed_time

def get_game_info(id):
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={id}"
    print(url)
    headers = {'User-Agent': 'Mozilla/5.0'}  # using a common user agent header

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    teams = []
    home_away_list = [None, None]
    for i in data.get("boxscore").get("teams"):
        team_id = i.get("team").get("id")
        abbreviation = i.get("team").get("abbreviation")
        home_or_away = None
        for j in data.get("header").get("competitions")[0].get("competitors"):
            if j.get("id") == team_id:
                home_or_away = j.get("homeAway")
                if (home_or_away == "home"):
                    home_away_list[0] = abbreviation
                else:
                    home_away_list[1] = abbreviation
                break
        teams.append({"id": team_id, "abbreviation": abbreviation, "homeAway": home_or_away})
    teams = {teams[0].get("id"): {"abbreviation": teams[0].get("abbreviation"), "homeAway": teams[0].get("homeAway")},
             teams[1].get("id"): {"abbreviation": teams[1].get("abbreviation"), "homeAway": teams[1].get("homeAway")}}

    plays = []

    play_counter = 0
    for drive in data.get("drives").get("previous"):
    # drive=data.get("drives").get("previous")[0]
        # end_clock = drive.get("end").get("clock").get("displayValue")
        for index, play in enumerate(drive.get("plays")):
            type = play.get("type").get("text")
            text = play.get("text")
            if ("END QUARTER" in text or "END GAME" in text):
                continue
            clock = play.get("clock").get("displayValue")
            # if (index < len(drive.get("plays")) - 1):
            #     play_time = find_clock_elapsed_time(clock, drive.get("plays")[index + 1].get("clock").get("displayValue"))
            # else:
            #     try:
            #         clock = re.split(r'(\(|\))', text)[2]
            #     except IndexError:
            #         print(text)
            #         clock = re.split(r'(\(|\))', text)[2]
            #     play_time = find_clock_elapsed_time(clock, end_clock)
            home_score = play.get("homeScore")
            away_score = play.get("awayScore")
            period = play.get("period").get("number")
            yards = play.get("statYardage")
            possession = play.get("possession")
            if (play.get("start").get("down") == 0):
                start = {"down": play.get("start").get("down"), "yardLine": play.get("start").get("yardLine"), "team": teams.get(play.get("start").get("team").get("id")).get("abbreviation")}
            else:
                start = {"down": play.get("start").get("down"), "downDistanceText": play.get("start").get("downDistanceText"), "yardLine": play.get("start").get("yardLine"), "team": teams.get(play.get("start").get("team").get("id")).get("abbreviation")}

            end = {"down": play.get("end").get("down"), "downDistanceText": play.get("end").get("downDistanceText"), "yardLine": play.get("end").get("yardLine"), "team": teams.get(play.get("end").get("team").get("id")).get("abbreviation")}
            plays.append({"type": type, "clock": clock, "text": text, "home_score": home_score, "away_score": away_score, "period": period, "yards": yards, "possession": possession, "start": start, "end": end, "playCounter": play_counter})
            play_counter += 1

    return {"home": home_away_list[0], "away": home_away_list[1], "plays": plays}

def save_game_info(date, teams, file_name, indent=0):
    game_info = get_game_info(get_game_id(date, teams))
    with open(file_name, "w") as file:
        json.dump(game_info, file, indent)

if __name__ == '__main__':
    date = "20221030"
    teams = ["WSH", "IND"]
    game_info = get_game_info(get_game_id(date, teams))
    # for i in game_info:
    #     if not i == "plays":
    #         print(f"{i}: {game_info.get(i)}")
    #     else:
    #         print("plays: ")
    #         for j in game_info.get(i):
    #             print(j)
    with open("game_info.json", "w") as file:
        json.dump(game_info, file, indent=4)