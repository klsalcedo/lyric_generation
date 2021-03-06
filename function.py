from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import config

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(config.spotify['client_id'],
                                                                        config.spotify['client_secret']))

def get_album_songs(uri_info):
    uri = []
    track = []
    duration = []
    explicit = []
    track_number = []
    one = sp.album_tracks(uri_info, limit=50, offset=0, market='US')
    df1 = pd.DataFrame(one)
    
    for i, x in df1['items'].items():
        uri.append(x['uri'])
        track.append(x['name'])
        duration.append(x['duration_ms'])
        explicit.append(x['explicit'])
        track_number.append(x['track_number'])
    
    df2 = pd.DataFrame({
    'uri':uri,
    'track':track,
    'duration_ms':duration,
    'explicit':explicit,
    'track_number':track_number})
    
    return df2

def get_track_info(df):
    danceability = []
    energy = []
    key = []
    loudness = []
    speechiness = []
    acousticness = []
    instrumentalness = []
    liveness = []
    valence = []
    tempo = []
    
    for i in df['uri']:
        for x in sp.audio_features(tracks=[i]):
            danceability.append(x['danceability'])
            energy.append(x['energy'])
            key.append(x['key'])
            loudness.append(x['loudness'])
            speechiness.append(x['speechiness'])
            acousticness.append(x['acousticness'])
            instrumentalness.append(x['instrumentalness'])
            liveness.append(x['liveness'])
            valence.append(x['valence'])
            tempo.append(x['tempo'])
            
    df2 = pd.DataFrame({
    'danceability':danceability,
    'energy':energy,
    'key':key,
    'loudness':loudness,
    'speechiness':speechiness,
    'acousticness':acousticness,
    'instrumentalness':instrumentalness,
    'liveness':liveness,
    'valence':valence,
    'tempo':tempo})
    
    return df2

def popularity(df):
    empty = []
    for i in df['uri']:
            series_track = pd.Series(sp.track(i))
            empty.append(series_track)
    df2 = pd.DataFrame(empty)
    return df2

def add_popularity(df):
    pop = popularity(df)
    df['popularity'] = pop['popularity']
    return df.head()

def lyrics_to_df(data, df):
    for i in range(len(data['tracks'])):
        album = data['name']
        title = data['tracks'][i]['song']['title']
        lyric = data['tracks'][i]['song']['lyrics']
        df = df.append({'track': title,'album': album, 'lyrics': lyric}, ignore_index=True)
    return df

def single_reg(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)
    print('Train Root Mean Square Error:', train_mse**0.5)
    print('Test Root Mean Square Error:', test_mse**0.5)
    
    return model

def log_transform(x):
    x = x+1
    return np.log(x)
transformer = FunctionTransformer(log_transform)


def full_reg(model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([('ss', StandardScaler()), ('regressor', model)])

    pipeline.fit(X_train, y_train)
    y_hat_train = pipeline.predict(X_train)
    y_hat_test = pipeline.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)
    print('Train Root Mean Square Error:', train_mse**0.5)
    print('Test Root Mean Square Error:', test_mse**0.5)
    
    return model