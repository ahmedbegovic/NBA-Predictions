import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from nba_api.stats.static import players
import matplotlib.pyplot as plt

#Load the data
df = pd.read_csv('stats/player_stats.csv')

# Keep the player names for identification on the plot screen
player_ids = df['Player_ID']
player_ids = player_ids.astype(int)

# Get active players for the graph
active_players = players.get_active_players()

# Keep the rest of the features for clustering options
features = df.drop('Player_ID', axis="columns")

#Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

#Clustering operations
kmeans = KMeans(n_clusters = 5, random_state = 26) 
df['Cluster'] = kmeans.fit_predict(scaled_features)

#Two statistics for the plot
statistic_1 = 'FGA'
statistic_2 = 'PTS'

#Draw a plot with the two chosen statistics, in this case field-goal attempts on the x-axis and points on the y-axis
figure, axes = plt.subplots(figsize=(12, 8))
scatter = axes.scatter(df[statistic_1], df[statistic_2], c=df['Cluster'])

# Add labels for each point - Code found here: https://stackoverflow.com/questions/7908636/how-to-add-hovering-annotations-to-a-plot
annot = axes.annotate("", xy=(0, 0), xytext=(20, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    position = scatter.get_offsets()[ind["ind"][0]]
    annot.xy = position
    #Find the player on the coordinate
    player_id = player_ids.iloc[ind['ind'][0]]
    #Use player api endpoint to get the full name
    player_name = players.find_player_by_id(player_id)
    text = player_name["full_name"]
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(1)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == axes:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            figure.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                figure.canvas.draw_idle()

figure.canvas.mpl_connect("motion_notify_event", hover)

plt.title('Player Clustering')
plt.xlabel(statistic_1)
plt.ylabel(statistic_2)
plt.grid(True)
plt.show()