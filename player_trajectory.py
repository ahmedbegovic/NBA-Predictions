import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the previous season data for Lebron and the newer seasons data for LeBron
player_data = pd.read_csv("stats/old_lebron.csv")
target_data = pd.read_csv("stats/new_lebron.csv")

# I want to predict the current season points based on the previous season points
X = player_data[['PTS', 'REB', 'AST']].values
y = target_data[['PTS']].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 26)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs = 200, batch_size = 16, validation_split = 0.2)

# Evaluate the model and the training loss
loss = model.evaluate(X_test_scaled, y_test)
print("Training Loss:", loss)

# Predict LeBron's future points per game
lebron_stats = np.array([[25, 5, 3]])  # Previous season stats: points, rebounds, assists
scaled_lebron_stats = scaler.transform(lebron_stats)
predicted_average = model.predict(scaled_lebron_stats)
print("Predicted average points per game:", predicted_average)
