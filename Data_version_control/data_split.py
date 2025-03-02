import os
import json

with open('./Data_version_control/config.json','r') as f:
    config_json = json.load(f)
    
local_data_dir = config_json['LOCAL_DATA_DIR']

if not os.path.exists(local_data_dir):
    os.mkdir(local_data_dir)
else:
    pass
    
def load_data():
    import pandas as pd
    DATA_PATH=os.getenv("DATA_DIR")
    spotify_df = pd.read_csv(os.path.join(DATA_PATH, 'Spotify/spotify_songs.csv'))
    x_columns = ['playlist_genre', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    y_column = 'track_popularity'
    X = spotify_df[x_columns]
    y = spotify_df[y_column]
    return X,y




def train_test():
    from sklearn.model_selection import train_test_split
    X,y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    print("Сургалтын X-н хэмжээ:", X_train.shape, "Y-н хэмжээ:", y_train.shape)
    print("Тестийн X-н хэмжээ:", X_test.shape, "Y-н хэмжээ:", y_test.shape)
    # return X_train, X_test, y_train, y_test
    X_train.to_csv(os.path.join(local_data_dir,'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(local_data_dir,'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(local_data_dir,'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(local_data_dir,'y_test.csv'), index=False)

if __name__ == "__main__":
    train_test()