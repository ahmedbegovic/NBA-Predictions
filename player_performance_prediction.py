import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load stats of the player, in this case it's Lebron James
df = pd.read_csv('stats/lebron.csv')

# Drop features that are categorical
df = df.drop(columns=["SEASON_ID", "Player_ID", "Game_ID", "GAME_DATE", "MATCHUP", "WL", "VIDEO_AVAILABLE"])

# Features that are going to be used to train are field goals missed, free throws missed, rebounds, assists, steals and blocks
# The reason for that is data focuses on last five seasons of NBA and the game is more focused on shots along with standard stats
X = df[["FGM", "FTM", 'REB', 'AST', 'STL', 'BLK']]
y = df["PTS"]

X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y, test_size=0.2, random_state=26)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error is :", mse)

# Example
example = np.array([[11, 8, 7, 2, 1, 1]])
predicted_points = model.predict(example)
print("Predicted points are:", predicted_points[0])

#Output is:
#Mean squared error is : 2.9886033458451933
#Predicted points are: 32.50152308095383