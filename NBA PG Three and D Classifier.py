"""
Description:
Classifies NBA PGs as a 3/D Player (1) or not a 3/D Player (0)
"""
import pandas as pd
import numpy as np
import math
import random

def distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i]-point2[i])**2
    distance = distance**0.5
    return distance

def classify(unknown, training_set, training_labels, k):
    distances = []
    for point in training_set:
        distance_to_point = distance(point[1:], unknown[1:])
        distances.append([distance_to_point, point[0]])
    distances.sort()
    neighbors = distances[0:k]
    num_true = 0
    num_false = 0
    for neighbor in neighbors:
        player = neighbor[1]
        if training_labels[player] == 0:
            num_false += 1
        elif training_labels[player] == 1:
            num_true += 1
    if num_true > num_false:
        return 1
    else:
        return 0

def validation_accuracy(training_set, training_labels, validation_set, validation_labels, k):
    num_correct = 0.0
    for point in validation_set:
        guess = classify(point, training_set, training_labels, k)
        if (guess == validation_labels[point[0]]):
            num_correct += 1
    return num_correct/len(validation_set)

df = pd.read_csv('NBA_Player_Data.csv')
df_advanced = pd.read_csv('NBA_Player_Advanced_Data.csv')
temp_df = pd.read_csv('NBA_Player_Data.csv')
temp_df = temp_df.drop(columns=['Rk', 'Player', 'Tm', 'Pos', 'Age'])

df_advanced = df_advanced.drop(columns = ['Rk', 'Pos', 'Age', 'Tm' , 'G', 'MP', 'PER', 'Unnamed: 19', 'Unnamed: 24', 'USG%', 'WS', 'OWS', 'WS/48', 'OBPM', 'BPM', 'VORP'])
df_advanced = df_advanced.drop([0])

drop = []
i = 0
for index, row in temp_df.iterrows():
	for columnName, columnData in temp_df.iteritems():
		if math.isnan(row[columnName]):
			drop.append(i)
	i += 1
df = df.drop(drop)
df = df.drop(columns = ['Rk', 'Tm', 'Pos','Age', 'G', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'FT', 'FTA'])

df = pd.merge(df, df_advanced)

i = 0
player_list=[]
while i < len(df['Player']):
    player_list.append(df['Player'][i])
    i+=1
i = 0
while i < len(player_list):
    player_list[i]=player_list[i].split('\\')
    player_list[i]=player_list[i][0]
    i+=1
df.Player=player_list

ThreeP_PCT = df['3P%'].tolist()
ThreePAr = df['3PAr'].tolist()
DWS = df['DWS'].tolist()
DBPM = df['DBPM'].tolist()

ThreeP_PCT_60PCT = np.quantile(ThreeP_PCT, 0.60)
ThreePAr_60PCT = np.quantile(ThreePAr, 0.60)
DWS_3Q = np.quantile(DWS, 0.75)
DBPM_3Q = np.quantile(DBPM, 0.75)

ThreeP_PCT_Med = np.quantile(ThreeP_PCT, 0.50)
ThreePAr_Med = np.quantile(ThreePAr, 0.50)
DWS_Med = np.quantile(DWS, 0.50)
DBPM_Med = np.quantile(DBPM, 0.50)

definite_three_and_d_players = []
for index, row in df.iterrows():
    if (row['3P%'] >= ThreeP_PCT_60PCT and 
        row['3PAr'] >= ThreePAr_60PCT and 
        row['DWS'] >= DWS_3Q and
        row['DBPM'] >= DBPM_3Q):
        definite_three_and_d_players.append(row['Player'])
random.shuffle(definite_three_and_d_players)
        
definite_not_three_and_d_players = []
i = 0
for index, row in df.iterrows():
    if (row['3P%'] <= ThreeP_PCT_Med and
        row['3PAr'] <= ThreePAr_Med and
        row['DWS'] <= DWS_Med and
        row['DBPM'] <= DBPM_Med):
        definite_not_three_and_d_players.append(row['Player'])
        
random.shuffle(definite_not_three_and_d_players)
definite_not_three_and_d_players = definite_not_three_and_d_players[1:14]

        
training_set = []
validation_set = []
training_labels = {}
validation_labels = {}    
add3andDPlayersval = ['Trevor Ariza', 'Khris Middleton', 'Wesley Matthews']    
addnon3andDPlayersval = ['Russell Westbrook', "De'Aaron Fox", 'Collin Sexton']
add3andDPlayerstrain = ['Kevin Durant', 'Bogdan Bogdanović', 'P.J. Tucker', 'Jerami Grant']
addnon3andDPlayerstrain = ['Pascal Siakam', 'Jimmy Butler', 'Buddy Hield', 'Kelly Oubre Jr.']
for i in range(len(definite_three_and_d_players)-2):
    point = []
    for columnNames in df[df['Player'] == definite_three_and_d_players[i]]:
        point.append(df[df['Player'] == definite_three_and_d_players[i]][columnNames].tolist())
    training_set.append(point)
for i in range(len(definite_three_and_d_players)-2, len(definite_three_and_d_players)):
    point = []
    for columnNames in df[df['Player'] == definite_three_and_d_players[i]]:
        point.append(df[df['Player'] == definite_three_and_d_players[i]][columnNames].tolist())
    validation_set.append(point)    

for player in add3andDPlayersval:
    point = []
    for columnNames in df[df['Player'] == player]:
        point.append(df[df['Player'] == player][columnNames].tolist())
    validation_set.append(point)
    
for player in add3andDPlayerstrain:
    point = []
    for columnNames in df[df['Player'] == player]:
        point.append(df[df['Player'] == player][columnNames].tolist())
    training_set.append(point)
    
for i in range(len(training_set)):
    training_labels[training_set[i][0][0]] = 1
for i in range(len(validation_set)):
    validation_labels[validation_set[i][0][0]] = 1

positive_length_training_set = len(training_set)
positive_length_validation_set = len(validation_set)
    
for i in range(len(definite_not_three_and_d_players)-2):
    point = []
    for columnNames in df[df['Player'] == definite_not_three_and_d_players[i]]:
        point.append(df[df['Player'] == definite_not_three_and_d_players[i]][columnNames].tolist())
    training_set.append(point)
for i in range(len(definite_not_three_and_d_players)-2, len(definite_not_three_and_d_players)):
    point = []
    for columnNames in df[df['Player'] == definite_not_three_and_d_players[i]]:
        point.append(df[df['Player'] == definite_not_three_and_d_players[i]][columnNames].tolist())
    validation_set.append(point)
    
for player in addnon3andDPlayersval:
    point = []
    for columnNames in df[df['Player'] == player]:
        point.append(df[df['Player'] == player][columnNames].tolist())
    validation_set.append(point)

for player in addnon3andDPlayerstrain:
    point = []
    for columnNames in df[df['Player'] == player]:
        point.append(df[df['Player'] == player][columnNames].tolist())
    training_set.append(point)
    
for i in range(positive_length_training_set, len(training_set)):
    training_labels[training_set[i][0][0]] = 0
for i in range(positive_length_validation_set, len(validation_set)):
    validation_labels[validation_set[i][0][0]] = 0

formatted_training_set = []
formatted_validation_set = []
i = 0
while i < len(training_set):
    j = 0
    temp_list = []
    while j < len(training_set[i]):
        temp_list.append(training_set[i][j][0])
        j += 1
    formatted_training_set.append(temp_list)
    i += 1
i = 0
while i < len(validation_set):
    j = 0
    temp_list = []
    while j < len(validation_set[i]):
        temp_list.append(validation_set[i][j][0])
        j += 1
    formatted_validation_set.append(temp_list)
    i += 1
training_set = formatted_training_set
validation_set = formatted_validation_set

inputted_player = input('Input an NBA player: ')
inputted_point = []
for columnNames in df[df['Player'] == inputted_player]:
    inputted_point.append(df[df['Player'] == inputted_player][columnNames].tolist())
formatted_inputted_point = []            
i = 0
while i < len(inputted_point):
    formatted_inputted_point.append(inputted_point[i][0])
    i += 1
inputted_point = formatted_inputted_point

accuracies = []
k = 1
while k <= len(training_set):
    accuracies.append(validation_accuracy(training_set, training_labels, validation_set, validation_labels, k))
    k += 1

ideal_k = 0
largest_acc = 0
i = 0
while i < len(accuracies):
    if accuracies[i] >= largest_acc:
        largest_acc = accuracies[i]
        ideal_k = i+1
    i += 1
    

print(classify(inputted_point, training_set, training_labels, ideal_k))