import pandas as pd
from sklearn.model_selection import train_test_split
import os
 
class LoadSplitData:
    
    def __init__(self, config, test_size=0.3):
        self.config = config
        self.test_size = test_size
        pass
    def load_data(self):
        
        DATA_PATH=self.config["DATA_DIR"]
        self.spotify_df = pd.read_csv(os.path.join(DATA_PATH, 'Spotify/spotify_songs.csv'))
        # x_columns = ['playlist_genre', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        #     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        # y_column = 'track_popularity'
        # X = spotify_df[x_columns]
        # y = spotify_df[y_column]
        # return X,y
        print("Өгөгдлийг импортлосон")

    def train_test(X,y):
       
        X_train, X_test, y_train, y_test = train_test_split(
           self.spotify_df[self.config["DATA_COLUMNS"]["X_COLUMNS"]], 
           self.spotify_df[self.config["DATA_COLUMNS"]["Y_COLUMN"]], test_size= self.test_size, random_state=123)
        print("Сургалтын X-н хэмжээ:", X_train.shape, "Y-н хэмжээ:", y_train.shape)
        print("Тестийн X-н хэмжээ:", X_test.shape, "Y-н хэмжээ:", y_test.shape)
        return X_train, X_test, y_train, y_test
