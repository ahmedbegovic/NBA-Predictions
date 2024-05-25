import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

#Load the data (all the games from the last five seasons)
df = pd.read_csv('stats/team_games.csv')

# Drop features that are categorical and reduntant
df = df.drop(columns=["SEASON_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ABBREVIATION", "TEAM_NAME", "MIN"])

#Since Win/Loss feature is categorized as W or L in the data, it's going to be changed to 1 and 0 respectively.
df['WL']= df['WL'].replace({'L': 0, 'W': 1})

# Fill missing values with the mean of each column
df.fillna(df.mean(), inplace=True)

X = df.drop(columns=['WL'])
y = df['WL']

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y, test_size = 0.2, random_state = 42)

#Train the model
model = LogisticRegression(max_iter = 200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error is :", mse)

#Example prediction for a game
example = np.array([[120, 30, 80, 0.375, 10, 25, 0.400, 8, 24, 0.333, 11, 24, 35, 30, 3, 4, 8, 20, 18.0]])
predicted_outcome = model.predict(example)

if(predicted_outcome[0] == 1.0):
    print("Predicted outcome: Win for the team")
else:
    print("Predicted outcome: Loss for the team")
