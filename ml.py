from scipy.io.wavfile import read
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import numpy as np
import librosa
#import torch
import os
print("*******SUCCESSFUL IMPORT")
#load dataset
dataset = "dataset/splitted/"
num_labels = 10

labels = []
audios = []
for label in range(num_labels):
    label_path = f"{dataset}/{label}"
    for file in sorted(os.listdir(label_path)):
        file_path = label_path + "/" + file
        sample_rate, audio = read(file_path)
        labels.append(label)
        audios.append(audio)
labels = np.array(labels)
#prepare features
max_duration_sec = 0.6
max_duration = int(max_duration_sec * sample_rate + 1e-6)
features = []

features_flatten = []
for audio in audios:
    if len(audio) < max_duration:
        audio = np.pad(audio, (0, max_duration - len(audio)), mode='constant')
    feature = librosa.feature.melspectrogram(audio.astype(float), sample_rate, n_mels=16, fmax=4000)
    features.append(feature)
    features_arr = np.array(features)
    features_flatten.append(feature.reshape(-1))
d2_features_arr = features_arr.reshape((289, 16*57)) #these numbers are taken from features_arr.shape 

print([feature.shape for feature in features])

def plot(idx):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"{labels[idx]}")
    plt.plot(audios[idx])

    plt.subplot(1, 2, 2)
    plt.title(f"{labels[idx]}")
    plt.imshow(features[idx])

    display(Audio(audios[idx], rate=sample_rate))
plot(4)
#splitting
features_train, features_test, labels_train, labels_test = train_test_split(d2_features_arr, labels, test_size=0.20, random_state=42, stratify = labels)
#train model
model = RandomForestClassifier(max_depth=30, n_estimators=30, max_features=7)
#model = MLPClassifier(hidden_layer_sizes=(800,400,200,150))
model.fit(features_train, labels_train)
#validate model
labels_test_predicted = model.predict(features_test)
print((labels_test_predicted == labels_test).mean())
print(labels_test_predicted)
print(labels_test)
#SAVE MODEL
import pickle
filename = "model.pkl"
model_pickled = pickle.dumps(model)
with open(filename, 'wb') as f:
    f.write(model_pickled)

