import numpy as np
import librosa
import librosa.feature
import pandas as pd
import os

import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, RandomizedSearchCV # Split de dataset et optimisation des hyperparamètres
from sklearn.ensemble import RandomForestClassifier # Random forest
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, zero_one_loss, classification_report # Métriques pour la mesure de performances
from sklearn.preprocessing import normalize, StandardScaler

import seaborn as sns
import joblib

genres = ["classical", "jazz", "pop", "reggae", "rock"]

def csv_creation():
    # retrieving all wav files form 5 different genres (jazz 54 missing, maybe corrupted)
    
    wav_files = {}

    for g in genres:
        wav_files[g] = []
        dir_path = "./Data/genres_original/" + g
        for file in os.listdir(dir_path):
            file_path = dir_path + "/" + file
            wav_files[g].append(librosa.load(file_path)[0])

    column_names = ['zcr', 'spectral_c', 'tempo', 'mfcc1', 'mfcc2', 'mfcc3',
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
    
    df.to_csv('music.csv', index = False)
    return df


def print_param():
    rf = RandomForestClassifier(random_state = 0)
    print('Parameters currently in use:\n')
    print(rf.get_params())


def random_forest():
    df= pd.read_csv('music.csv')
    features = df
    # valeurs à prédire
    labels = np.array(features['label'])
    # supprime les labels des données
    features = features.drop('label', axis = 1)
    # sauvegarde le nom de features
    feature_list = list(features.columns)
    # conversion en numpy array
    features = np.array(features)

    # séparer les données en training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.35, random_state = 0)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    sc = StandardScaler()
    train_features = sc.fit_transform(train_features)
    test_features = sc.transform(test_features)

    # nombre d'arbres
    n_estimators = [2000, 4000]
    # profondeur max de l'arbre
    max_depth = [20]
    max_depth.append(None)
    # nombre d'échantillon min nécessaire par noeuds
    min_samples_split = [2, 4]#[2]
    # nombre d'échantillon min nécessaire par feuilles
    min_samples_leaf = [1, 2]#[1]

    # création de la grille
    random_grid = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                }
    print(random_grid)

    # création du modèle
    rf = RandomForestClassifier(n_estimators=4000, max_features='sqrt', max_depth=20, min_samples_split=2, min_samples_leaf=1, bootstrap=True, criterion='gini' ,random_state=0)

    # fit le modèle
    rf.fit(train_features, train_labels)

    # prédictions
    predictions = rf.predict(test_features)

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

    joblib.dump(rf, "model.pkl")

def predict_song(filename):
    model = joblib.load("model.pkl")
    music = librosa.load(filename)[0]
    print(music)
    classes = model.classes_.tolist()

    time = 0
    total_duration = 30
    frequency = 6 # between 6 and 7 to have approximatively 1 second worth of computing done in 1 real second
    nb_samples = 1
    sr = 22050
    features = [0 for i in range(23)]
    sum_features = [0 for i in range(22)] # no need to stock zero crossings sum, already in features array

    while (time < total_duration):
        #print("1 : ", int(time*sr) , "2 : ", int((time + 1/frequency) * sr))
        sub_music = music[int(time*sr):int((time + 1/frequency) * sr)]
        zcr = librosa.zero_crossings(sub_music)
        features[0] = np.add(features[0], (sum(zcr)))

        spectral_centroid = librosa.feature.spectral_centroid(y = sub_music, n_fft=256)[0]
        sum_features[0] += np.mean(spectral_centroid)
        features[1] = sum_features[0] / nb_samples

        tempo =librosa.feature.tempo(y = sub_music)
        sum_features[1] += np.mean(tempo)
        features[2] = sum_features[1] / nb_samples

        mfcc = librosa.feature.mfcc(y = sub_music, n_mels = 15, n_fft = 256)
        for i in range(len(mfcc)):
            sum_features[2+i] += np.mean(mfcc[i])
            features[3 + i] = sum_features[2+i] / nb_samples

        output = model.predict_proba([features])
        output = output.tolist()[0]
        prediction = classes[output.index(max(output))]
        time += 1/frequency
        nb_samples += 1
        print(time, ": predicted", prediction, "with probabilities : \n", classes[0] , "->", output[0], "\n", classes[1] , "->", output[1], "\n", classes[2] , "->", output[2], "\n", classes[3] , "->", output[3], "\n", classes[4] , "->", output[4])
    print("Final prediction :", prediction, "with probability", max(output))
   

#df = csv_creation()
#random_forest()
res = predict_song("./Data/genres_original/classical/classical.00002.wav")