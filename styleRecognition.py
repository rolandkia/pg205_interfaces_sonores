import numpy as np
import librosa
import librosa.feature
import pandas as pd
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split # Split de dataset et optimisation des hyperparamètres
from sklearn.ensemble import RandomForestClassifier # Random forest
from sklearn.metrics import accuracy_score, confusion_matrix, zero_one_loss, classification_report # Métriques pour la mesure de performances
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import joblib

genres = ['classical', 'jazz', 'pop', 'reggae', 'rock']

def csv_creation_default_features():
    # retrieving all wav files form 5 different genres (jazz 54 missing, maybe corrupted)
    
    wav_files = {}

    for g in genres:
        wav_files[g] = []
        dir_path = './Data/genres_original/' + g
        for file in os.listdir(dir_path):
            file_path = dir_path + '/' + file
            wav_files[g].append(librosa.load(file_path)[0])

    column_names = ['zcr', 'spectral_centroid', 'tempo', 'mfcc1', 'mfcc2', 'mfcc3',
                'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
                'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
                'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'label']
    df = pd.DataFrame(columns = column_names)
    count = 0

    # extraction of the features
    for g in genres:
        for music in wav_files[g]:
            features = []

            zcr = librosa.zero_crossings(music)
            features.append(sum(zcr))

            spectral_centroid = librosa.feature.spectral_centroid(y = music)[0]
            features.append(np.mean(spectral_centroid))

            tempo =librosa.feature.tempo(y = music)
            features.append(np.mean(tempo))

            mfcc = librosa.feature.mfcc(y = music)
            for m in mfcc:
                features.append(np.mean(m))

            df.loc[count] = features+[g]
            count += 1
    
    df.to_csv('csv/music_default_features.csv', index = False)
    return df


def csv_creation_with_contrast():
    # retrieving all wav files form 5 different genres (jazz 54 missing, maybe corrupted)
    
    wav_files = {}

    for g in genres:
        wav_files[g] = []
        dir_path = './Data/genres_original/' + g
        for file in os.listdir(dir_path):
            file_path = dir_path + '/' + file
            wav_files[g].append(librosa.load(file_path)[0])

    column_names = ['zcr', 'spectral_centroid', 'tempo', 'spectral_contrast', 'mfcc1', 'mfcc2', 'mfcc3',
                'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
                'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
                'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'label']
    df = pd.DataFrame(columns = column_names)
    count = 0

    # extraction of the features
    for g in genres:
        for music in wav_files[g]:
            features = []

            zcr = librosa.zero_crossings(music)
            features.append(sum(zcr))

            spectral_centroid = librosa.feature.spectral_centroid(y = music)[0]
            features.append(np.mean(spectral_centroid))

            tempo =librosa.feature.tempo(y = music)
            features.append(np.mean(tempo))

            spectral_contrast = librosa.feature.spectral_contrast(y = music)
            features.append(np.mean(spectral_contrast))

            mfcc = librosa.feature.mfcc(y = music)
            for m in mfcc:
                features.append(np.mean(m))

            df.loc[count] = features+[g]
            count += 1
    
    df.to_csv('csv/music_with_contrast.csv', index = False)
    return df

def csv_creation_without_zcr_tempo():
    # retrieving all wav files form 5 different genres (jazz 54 missing, maybe corrupted)
    wav_files = {}

    for g in genres:
        wav_files[g] = []
        dir_path = './Data/genres_original/' + g
        for file in os.listdir(dir_path):
            file_path = dir_path + '/' + file
            wav_files[g].append(librosa.load(file_path)[0])

    column_names = ['spectral_centroid', 'mfcc1', 'mfcc2', 'mfcc3',
                'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
                'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
                'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'label']
    df = pd.DataFrame(columns = column_names)
    count = 0

    # extraction of the features
    for g in genres:
        for music in wav_files[g]:
            features = []

            spectral_centroid = librosa.feature.spectral_centroid(y = music)[0]
            features.append(np.mean(spectral_centroid))

            mfcc = librosa.feature.mfcc(y = music)
            for m in mfcc:
                features.append(np.mean(m))

            df.loc[count] = features+[g]
            count += 1
    
    df.to_csv('csv/music_without_zcr_tempo.csv', index = False)
    return df

def random_forest(csv_filename_extension):
    df= pd.read_csv('csv/music_' + csv_filename_extension + '.csv')
    features = df
    # values to predict
    labels = np.array(features['label'])
    # deleting labels
    features = features.drop('label', axis = 1)
    features = np.array(features)

    # separating data
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.35)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    sc = StandardScaler()
    train_features = sc.fit_transform(train_features)
    test_features = sc.transform(test_features)

    # creating model
    rf = RandomForestClassifier(n_estimators=2500, max_features='sqrt', max_depth=15, min_samples_split=2, min_samples_leaf=1, bootstrap=True, criterion='gini', random_state=0, n_jobs = 4)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)

    print('Parameters currently in use:\n')
    print(rf.get_params())

    # Zero_one_loss error
    errors = zero_one_loss(test_labels, predictions, normalize=False)
    print('zero_one_loss error :', errors)

    # Accuracy Score
    accuracy_test = accuracy_score(test_labels, predictions)
    print('accuracy_score on test dataset :', accuracy_test)

    print(classification_report(predictions, test_labels))

    sns.set_theme()
    mat = confusion_matrix(test_labels, predictions)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=genres, yticklabels=genres)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

    joblib.dump(rf, 'models/model_test' + csv_filename_extension + '.pkl')

def predict_song(filename, model_filename_extension):
    model = joblib.load('models/model_' + model_filename_extension + '.pkl')
    music = librosa.load(filename)[0]
    classes = model.classes_.tolist()

    time = 0
    total_duration = 30
    frequency = 3 # between 2 and 3 to have approximatively 1 second worth of computing done in 1 real second
    nb_samples = 1
    sr = 22050

    nb_features = 23
    sum_adjustement = 0
    if (model_filename_extension == 'with_contrast'):
        nb_features += 1
    if (model_filename_extension == 'without_zcr_tempo'):
        nb_features -= 2
        sum_adjustement = 1
    features = [0 for i in range(nb_features)] # 23 default, 24 with contrast, 21 without zcr and tempo
    sum_features = [0 for i in range(nb_features-1+sum_adjustement)] # no need to stock zero crossings sum, already in features array

    while (time < total_duration):
        sub_music = music[int(time*sr):int((time + 1/frequency) * sr)]

        feat_offset = 0
        sum_offset = 0

        if (model_filename_extension != 'without_zcr_tempo'):
            zcr = librosa.zero_crossings(sub_music)
            features[0] = np.add(features[0], (sum(zcr)))
        else :
            feat_offset = -1
            sum_offset = 1

        spectral_centroid = librosa.feature.spectral_centroid(y = sub_music, n_fft=2048)[0]
        sum_features[0] += np.mean(spectral_centroid)
        features[1 + feat_offset] = sum_features[0] / nb_samples

        if (model_filename_extension != 'without_zcr_tempo'):
            tempo = librosa.feature.tempo(y = sub_music)
            sum_features[1] += np.mean(tempo)
            features[2] = sum_features[1] / nb_samples
        else:
            feat_offset -= 1

        if (model_filename_extension == 'with_contrast'):
            spectral_contrast = librosa.feature.spectral_contrast(y = sub_music)
            sum_features[2] += np.mean(spectral_contrast)
            features[3] = sum_features[2] / nb_samples
        else:
            feat_offset -= 1

        mfcc = librosa.feature.mfcc(y = sub_music, n_mels = 20, n_fft = 2048)
        for i in range(len(mfcc)):
            sum_features[3 + feat_offset + sum_offset + i] += np.mean(mfcc[i])
            features[4 + feat_offset + i] = sum_features[3 + feat_offset + sum_offset + i] / nb_samples

        output = model.predict_proba([features])
        output = output.tolist()[0]
        prediction = classes[output.index(max(output))]
        time += 1/frequency
        nb_samples += 1
        print(time, ': predicted', prediction, 'with probabilities : \n', classes[0] , '->', output[0], '\n', classes[1] , '->', output[1], '\n', classes[2] , '->', output[2], '\n', classes[3] , '->', output[3], '\n', classes[4] , '->', output[4])
    print('Final prediction :', prediction, 'with probability', max(output))

model_used = 'default_features'
nb_features = 23
if (model_used == 'with_contrast'):
    nb_features += 1
if (model_used == 'without_zcr_tempo'):
    nb_features -= 2
sum_features = [0 for i in range(nb_features-1)] # no need to stock zero crossings sum, already in features array

# how to use it :
# initialize time at 0, total_duration at 30, frequency at 2 or 3, nb_samples at 1
# filename is the wav file (for example './Data/genres_original/jazz/jazz.00008.wav')
# model_filename_extension is the model to use (for example 'default_features' for model_default_features.pkl)
# while (time < total_duration) loop calling this function
# at each iteration, increase : time += 1/frequency, nb_samples += 1
def predict_song_for_graphics(filename, model_filename_extension, sum_features, time, nb_samples):
    model = joblib.load('models/model_' + model_filename_extension + '.pkl')
    music = librosa.load(filename)[0]

    frequency = 3 # between 2 and 3 to have approximatively 1 second worth of computing done in 1 real second
    sr = 22050
    features = [0 for i in range(len(sum_features)+1)] # 23 default, 24 with contrast, 21 without zcr and tempo

    sub_music = music[int(time*sr):int((time + 1/frequency) * sr)]

    feat_offset = 0
    sum_offset = 0

    if (model_filename_extension != 'without_zcr_tempo'):
        zcr = librosa.zero_crossings(sub_music)
        features[0] = np.add(features[0], (sum(zcr)))
    else :
        feat_offset = -1
        sum_offset = 1

    spectral_centroid = librosa.feature.spectral_centroid(y = sub_music, n_fft=2048)[0]
    sum_features[0] += np.mean(spectral_centroid)
    features[1 + feat_offset] = sum_features[0] / nb_samples

    if (model_filename_extension != 'without_zcr_tempo'):
        tempo = librosa.feature.tempo(y = sub_music)
        sum_features[1] += np.mean(tempo)
        features[2] = sum_features[1] / nb_samples
    else:
        feat_offset -= 1

    if (model_filename_extension == 'with_contrast'):
        spectral_contrast = librosa.feature.spectral_contrast(y = sub_music)
        sum_features[2] += np.mean(spectral_contrast)
        features[3] = sum_features[2] / nb_samples
    else:
        feat_offset -= 1

    mfcc = librosa.feature.mfcc(y = sub_music, n_mels = 20, n_fft = 2048)
    for i in range(len(mfcc)):
        sum_features[3 + feat_offset + sum_offset + i] += np.mean(mfcc[i])
        features[4 + feat_offset + i] = sum_features[3 + feat_offset + sum_offset + i] / nb_samples

    output = model.predict_proba([features])
    output = output.tolist()[0]
    return output, sum_features

def predict_song_from_mic(mic_song, model_filename_extension, sum_features, nb_samples):
    model = joblib.load('models/model_' + model_filename_extension + '.pkl')
    features = [0 for i in range(len(sum_features)+1)] # 23 default, 24 with contrast, 21 without zcr and tempo

    sub_music = mic_song

    feat_offset = 0
    sum_offset = 0

    if (model_filename_extension != 'without_zcr_tempo'):
        zcr = librosa.zero_crossings(sub_music)
        features[0] = np.add(features[0], (sum(zcr)))
    else :
        feat_offset = -1
        sum_offset = 1

    spectral_centroid = librosa.feature.spectral_centroid(y = sub_music, n_fft=2048)[0]
    sum_features[0] += np.mean(spectral_centroid)
    features[1 + feat_offset] = sum_features[0] / nb_samples

    if (model_filename_extension != 'without_zcr_tempo'):
        tempo = librosa.feature.tempo(y = sub_music)
        sum_features[1] += np.mean(tempo)
        features[2] = sum_features[1] / nb_samples
    else:
        feat_offset -= 1

    if (model_filename_extension == 'with_contrast'):
        spectral_contrast = librosa.feature.spectral_contrast(y = sub_music)
        sum_features[2] += np.mean(spectral_contrast)
        features[3] = sum_features[2] / nb_samples
    else:
        feat_offset -= 1

    mfcc = librosa.feature.mfcc(y = sub_music, n_mels = 20, n_fft = 2048)
    for i in range(len(mfcc)):
        sum_features[3 + feat_offset + sum_offset + i] += np.mean(mfcc[i])
        features[4 + feat_offset + i] = sum_features[3 + feat_offset + sum_offset + i] / nb_samples

    output = model.predict_proba([features])
    output = output.tolist()[0]
    return output, sum_features

# df = csv_creation_without_zcr_tempo()
# random_forest('default_features')
res = predict_song('./Data/genres_original/pop/pop.00045.wav', 'with_contrast')