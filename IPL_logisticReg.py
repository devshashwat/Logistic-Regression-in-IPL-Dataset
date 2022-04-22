import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
from pip._vendor.distlib.compat import raw_input

data_path = '/Applications/IPL/'
match_df = pd.read_csv(data_path + 'Match.csv')
ball_df = pd.read_csv(data_path + 'Ball_by_Ball.csv')
playerMatch_df = pd.read_csv(data_path + 'Player_Match.csv')
player_df = pd.read_csv(data_path + 'Player.csv')
season_df = pd.read_csv(data_path + 'Season.csv')
team_df = pd.read_csv(data_path + 'Team.csv')

searchplayer = raw_input('Enter the name of the Player playing the last over: ')


# finding the Player ID for given player in database
def userinput_player_id():
    player_name = searchplayer
    player_df['index'] = player_df['Player_Name'].str.find(player_name)
    userinput_player_id = int(player_df[player_df['index'] == 0]['Player_Id'])
    return userinput_player_id


inputplayer_id = userinput_player_id()


# finding the most recent team_id of the player
def teamid_of_inputplayer():
    teamidlist_of_inputplayer = playerMatch_df[playerMatch_df['Player_Id'] == inputplayer_id]['Team_Id']
    teamid_of_inputplayer = int(teamidlist_of_inputplayer.iloc[-1])
    return teamid_of_inputplayer


# finding the most recent team name of the player
def team_of_inputplayer():
    teamidlist_of_inputplayer = playerMatch_df[playerMatch_df['Player_Id'] == inputplayer_id]['Team_Id']
    teamid_of_inputplayer = int(teamidlist_of_inputplayer.iloc[-1])
    team_of_inputplayer = str(team_df[team_df['Team_Id'] == teamid_of_inputplayer]['Team_Name'])
    return team_of_inputplayer


# finding the list of matches input player has played for his recent team
def list_of_matches():
    no_of_matches_played = playerMatch_df[((playerMatch_df['Player_Id'] == inputplayer_id)
                                           & (playerMatch_df['Team_Id'] == teamid_of_inputplayer()))]
    return no_of_matches_played


match_list = list_of_matches()


# finding the the list of matches the team had played in a second innings
def no_of_matches_in_second_innings():
    no_of_matches_innings_two = match_df[((match_df['Team_Name_Id'] == teamid_of_inputplayer()) &
                                          (match_df['Toss_Decision'] == 'field') &
                                          (match_df['Toss_Winner_Id'] == teamid_of_inputplayer())) |
                                         ((match_df['Team_Name_Id'] == teamid_of_inputplayer()) &
                                          (match_df['Toss_Decision'] == 'bat') &
                                          (match_df['Toss_Winner_Id'] != teamid_of_inputplayer()))]
    return no_of_matches_innings_two


second_inn = no_of_matches_in_second_innings()


# finding the list of the matches input player has played in the second innings
def second_inning_played():
    merge = pd.merge(second_inn, match_list, on='Match_Id')
    return merge


# finding the list of matches input player has played the last over in the second innings ball by ball
def last_over_played():
    last_over = ball_df[((ball_df['Innings_Id'] == 2) & (ball_df['Over_Id'] == 20)
                         & (ball_df['Striker_Id'] == inputplayer_id) &
                         (ball_df['Team_Batting_Id'] == teamid_of_inputplayer())) |
                        ((ball_df['Innings_Id'] == 2) & (ball_df['Over_Id'] == 20) &
                         (ball_df['Non_Striker_Id'] == inputplayer_id) &
                         (ball_df['Team_Batting_Id'] == teamid_of_inputplayer()))]
    return last_over


last_over_playing = last_over_played()

# reducing from ball by ball to matches and dismissals
last_over_playing = last_over_playing.drop_duplicates(subset='Match_Id', keep='last')


# finding the list of matches input player has played in last over of 2nd innings without getting dismissed
def last_no_out():
    no_out = last_over_playing[((last_over_playing['Player_dissimal_Id'] != inputplayer_id) &
                                (last_over_playing['Dissimal_Type'] != 'caught') &
                                (last_over_playing['Dissimal_Type'] != 'bowled') &
                                (last_over_playing['Dissimal_Type'] != 'lbw') &
                                (last_over_playing['Dissimal_Type'] != 'run out'))]
    return no_out


# finding the list of matches input player has played in the last over of 2nd innings without getting dismissed
no_out_list = last_no_out()


# finding list of matches from the list of not out matches which input player played last over in 2nd innings were won
def list_matches_won():
    intersected = pd.merge(match_df, match_list, how='inner')
    match_winner = intersected[intersected['Match_Winner_Id'] == teamid_of_inputplayer()]
    return match_winner


# reducing from ball by ball to matches and dismissals
last_over_playing = last_over_playing.drop_duplicates(subset='Match_Id', keep='last')

# list of only matches for recent team by the player
player_matches = match_list['Match_Id']
player_matches.reset_index(drop=True)
list_player_matches = player_matches.values.tolist()

# list of only matches in second innings played by player
matches_in_second_innings = second_inning_played()['Match_Id']
matches_in_second_innings.reset_index(drop=True)
matches_in_second_innings = matches_in_second_innings.astype(int)
list_match_second_inn = matches_in_second_innings.values.tolist()

# list of only matches in second innings played by player in last over
last_over_match = last_over_playing["Match_Id"]
last_over_match.reset_index(drop=True)
last_over_match = last_over_match.astype(int)
list_last_over_match = last_over_match.values.tolist()

# list of only matches in second innings played by player in last over without getting dismissed
not_dismissed = no_out_list['Match_Id']
not_dismissed.reset_index(drop=True)
not_dismissed = not_dismissed.astype(int)
list_not_dismissed = not_dismissed.values.tolist()

# list of matches player has won out of all the matches he played for his most recent team
player_won_match = list_matches_won()["Match_Id"]
player_won_match.reset_index(drop=True)
player_won_match = player_won_match.astype(int)
list_player_won_match = player_won_match.values.tolist()

# forming a new lists for logistic Regression in form of 1 or 0 (t being true and 0 being false)

# list of second inn played by player (0/1)
second_in_final = [*range(0, len(list_player_matches))]
for i in range(0, len(list_player_matches)):
    if list_player_matches[i] in list_match_second_inn:
        second_in_final[i] = 1
    else:
        second_in_final[i] = 0

# list of last over played by player (0/1)
last_over_final = [*range(0, len(list_player_matches))]
for i in range(0, len(list_player_matches)):
    if list_player_matches[i] in list_last_over_match:
        last_over_final[i] = 1
    else:
        last_over_final[i] = 0

# list of not out played by player in last over (0/1)
not_dismissed_final = [*range(0, len(list_player_matches))]
for i in range(0, len(list_player_matches)):
    if list_player_matches[i] in list_not_dismissed:
        not_dismissed_final[i] = 1
    else:
        not_dismissed_final[i] = 0

# list of matches won played by player (0/1)
won_final = [*range(0, len(list_player_matches))]
for i in range(0, len(list_player_matches)):
    if list_player_matches[i] in list_player_won_match:
        won_final[i] = 1
    else:
        won_final[i] = 0

# Forming all the new 0/1 list into a table of dataframe
final_table = {'Match_Id': list_player_matches, 'Second_Innings': second_in_final, 'Played_Last_Over': last_over_final,
               'Not_Out_Last_Over': not_dismissed_final, 'Matches_Won': won_final}
df_final = pd.DataFrame(final_table, columns=['Match_Id', 'Second_Innings', 'Played_Last_Over', 'Not_Out_Last_Over',
                                              'Matches_Won'])
#print(df_final.head())

# Initializing the logistical Regression
X = df_final[['Second_Innings', 'Played_Last_Over', 'Not_Out_Last_Over']]
y = df_final['Matches_Won']

# Splitting the test and train (train = 75%, test = 25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Applying the logistic regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

# displaying the accuracy
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
print(X_test)  # test dataset
print('Predicted Data: ', y_pred)  # predicted values

confusion_matrix = pd.crosstab(y_test,y_pred ,rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
plt.show()

