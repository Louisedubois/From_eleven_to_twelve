
###
# Copyright 2022 M Blazevic, Y Bouachera, L Dubois, E Luu, M Naour, F Telmini. All Rights Reserved.
# Licensed under the Creative Commons Attribution-NonCommercial International Public License, Version 4.0.
# You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/legalcode
###

#!/usr/bin/python
import warnings
warnings.simplefilter("ignore", UserWarning)
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import os
from pydub import AudioSegment
from spectralcluster import SpectralClusterer
import pandas as pd
import numpy as np
import shutil
import librosa
import pickle
from tqdm import tqdm

# !!!!!!!!!!! import sampling_rate, est ce que c'est le même pour notre base ???
from resemblyzer.audio import sampling_rate



def perform_diarization(path, encoder, min_clusters = 2, max_clusters = 10, sampling_rate = sampling_rate):

    audio_wav = preprocess_wav(path)
    _, cont_embeds, splitted_wav = encoder.embed_utterance(audio_wav, return_partials=True, rate=16)

    clusterer = SpectralClusterer(min_clusters, max_clusters)
    labels = clusterer.predict(cont_embeds)


    def create_labelling(labels, wav_splits, sampling_rate):
        times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
        labelling = []
        start_time = 0

        labels = clusterer.predict(cont_embeds)

        for i,time in enumerate(times):
            if i>0 and labels[i]!=labels[i-1]:
                temp = [str(labels[i-1]),start_time,time]
                labelling.append(tuple(temp))
                start_time = time
            if i==len(times)-1:
                temp = [str(labels[i]),start_time,time]
                labelling.append(tuple(temp))

        return labelling

    return path, create_labelling(labels, splitted_wav, sampling_rate)
    
    
def slice_audio(path, diarization):
    audio = AudioSegment.from_wav(path)

    output_dict = dict()

    for (speaker_id, start_time, stop_time) in diarization:
        audio_slice = audio[int(start_time*1000): int(stop_time*1000)]
        output_dict.setdefault("speaker_%s" % speaker_id, []).append(audio_slice)

    return output_dict
    
    
def get_features_slices(sliced_audio):
    features_dict = {}
    for speaker_id, audio_samples in sliced_audio.items():
        for audio_sample in audio_samples :
            audio_sample.export("temp_audio.wav", format="wav")
            x, sample_rate = librosa.load("temp_audio.wav")
            mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
            os.remove("temp_audio.wav")
        
            features_dict.setdefault(speaker_id, []).append({'sample_rate':sample_rate, 'mfccs':mfccs})
    return features_dict


def predict_from_slice(sliced_audios, model, labels = {0 : "F", 1 : "M"}):
    features_dict = get_features_slices(sliced_audios)

    result_dict = {}
    for speaker_id, samples_list in features_dict.items():
        for sample_audio in samples_list:
            prediction = model.predict(sample_audio['mfccs'].reshape(1,-1))
            result_dict.setdefault(speaker_id, []).append(labels[prediction[0]])
    return result_dict
    
def make_count_prediction (path, encoder, model, min_clusters):

    path, diarization = perform_diarization(path, encoder, min_clusters)
    sliced_audios = slice_audio(path, diarization)
    predicted = predict_from_slice(sliced_audios, model)
    count_speakers = {'F' : 0, 'M' : 0}

    for speaker_id, prediction_list in predicted.items():
        sex, counts = np.unique(prediction_list, return_counts = True)
        count_speakers[sex[np.argmax(counts)]] += 1

    return path, count_speakers

def make_all_preds(data_path, model_path = "ADAmodel.pkl", min_clusters = 2):
    
    model = pickle.load(open(model_path, 'rb'))
    
    encoder = VoiceEncoder("cpu")
    labels_raw = pd.read_excel(os.path.join(data_path, 'referential_movies_subtitles_with_gender.xlsx'))
    labels = labels_raw[['sound_clip_id', 'nb_gender_M', 'nb_gender_F', 'nb_gender_NA', 'nb_gender_NB']]

    results_df = pd.DataFrame(columns = ['sound_clip_id', 'y_M', 'y_F', 'y_NA', 'y_NB', 'pred_F', 'pref_M'])

    for index in tqdm(range(len(labels))):
        filename = str(labels.loc[index, 'sound_clip_id']) + '.wav'

        path, pred_count = make_count_prediction(os.path.join(data_path, filename), encoder, model, min_clusters)

        file_df = pd.DataFrame([[labels_raw.loc[index, 'sound_clip_id'], 
                                 labels_raw.loc[index, 'nb_gender_M'], 
                                 labels_raw.loc[index, 'nb_gender_F'], 
                                 labels_raw.loc[index, 'nb_gender_NA'],
                                 labels_raw.loc[index, 'nb_gender_NB'], 
                                pred_count['F'], 
                                pred_count['M']]], columns = results_df.columns)
        results_df = pd.concat([results_df, file_df], ignore_index = True)

    return results_df
    


