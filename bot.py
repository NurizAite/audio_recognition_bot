import sys          #argv
import numpy as np # generate random
import os          # get current directory path
import subprocess  # execute ffmpeg
import telebot     # run telegram bot
from scipy.io.wavfile import read, write
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import numpy as np
import librosa
#import torch
import os
from datetime import datetime # generate log
#Load model
import pickle
filename = "model.pkl"
with open(filename, 'rb') as f:
    model_pickled = f.read()
model = pickle.loads(model_pickled)
#MODEL_PICKELDKEJ END
#validate model
#labels_test_predicted = model.predict(features_test)
#print((labels_test_predicted == labels_test).mean())
#print(labels_test_predicted)
#print(labels_test)

#Smth went wrong (uninit values) and i added it
max_duration_sec = 0.6

#****BOT

users_tasks = dict()
bot = telebot.TeleBot('1236434003:AAG126MJpMR7TTEGOm4B2eBjbCPW2xKTEr4')
root = os.getcwd() + "/dataset/"


def save_ogg(ogg_data, ogg_path):
    with open(ogg_path, "wb") as file:
        file.write(ogg_data)


def convert_ogg_wav(ogg_path, dst_path):
    rate = 48000
    cmd = f"ffmpeg -i {ogg_path} -ar {rate} {dst_path} -y -loglevel panic"
    log(cmd)
    with subprocess.Popen(cmd.split()) as p:
        try:
            p.wait(timeout=2)
        except:
            p.kill()
            p.wait()
            return "timeout"


#def generate_task():
#    return ' '.join(list(map(str, np.random.randint(10, size=5))))
def log(text):
    time_stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    print(time_stamp + " " + text)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    user = message.from_user.id
    text = message.text
    log(f"User ({user}): {text}")
#    users_tasks[user] = generate_task()
#    bot.send_message(user,
#        f"Пожалуйста, назови мне любую цифру:\n{users_tasks[user]}")
    bot.send_message(user, "Назовите 5 цифр с паузами и я попробую их отгадать")

@bot.message_handler(content_types=['voice'])
def get_voice_messages(message):
    user = message.from_user.id
    voice = message.voice
#    if user not in users_tasks:
#        bot.send_message(user, "/start")
    log(f"User ({user}): voice")

    tele_file = bot.get_file(voice.file_id)
    ogg_data = bot.download_file(tele_file.file_path)
  #  file_name = users_tasks[user].replace(" ", "_")
    file_name = "0_0_0_0_0"
    ogg_path = root + "/request/ogg/" + file_name + ".ogg"
    wav_path = root + "/request/wav/" + file_name + ".wav"
    save_ogg(ogg_data, ogg_path)
    convert_ogg_wav(ogg_path, wav_path)
    split(user)

#SPLITTING OF REQUEST
def print_with_timeline(data, single_duration, units_name, row_limit):
    for i in range(len(data)):
        if i % row_limit == 0:
            print(f"{single_duration * i:8.3f} {units_name} |  ", end='')
        print(f"{data[i]:.3f} ", end='')
        if (i + 1) % row_limit == 0 or i + 1 == len(data):
            print(f" | {single_duration * (i + 1):8.3f} {units_name}")


def get_segment_energy(data, start, end):
    energy = 0
    for i in range(start, end):
        energy += float(data[i]) * data[i] / (end - start)
    energy = np.sqrt(energy) / 32768
    return energy


def get_segments_energy(data, segment_duration):
    segments_energy  = []
    for segment_start in range(0, len(data), segment_duration):
        segment_stop = min(segment_start + segment_duration, len(data))
        energy = get_segment_energy(data, segment_start, segment_stop)
        segments_energy.append(energy)
    return segments_energy


def get_vad_mask(data, threshold):
    vad_mask = np.zeros_like(data)
    for i in range(0, len(data)):
        vad_mask[i] = data[i] > threshold
    return vad_mask


def sec2samples(seconds, sample_rate):
  return int(seconds * sample_rate)


class Segment:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop


def print_segments(segments, single_duration, units_name):
    total_duration = 0.0
    min_duration = 0.0
    max_duration = 0.0
    for i in range(len(segments)):
        start_units = segments[i].start * single_duration
        stop_units = segments[i].stop * single_duration
        duration_units = stop_units - start_units
        total_duration += duration_units
        if i == 0 or min_duration > duration_units:
            min_duration = duration_units
        if i == 0 or max_duration < duration_units:
            max_duration = duration_units
        print(f"{i:5}: {start_units:6.3f} - {stop_units:6.3f} ({duration_units:6.3f} {units_name})")
    print(f"Min   duration: {min_duration:.3f} {units_name}")
    print(f"Mean  duration: {total_duration / len(segments):.3f} {units_name}")
    print(f"Max   duration: {max_duration:.3f} {units_name}")
    print(f"Total segments: {len(segments)}")
    print(f"Total duration: {total_duration:.3f} {units_name}")


def mask_compress(data):
    segments = [];
    if len(data) == 0:
        return segments
    start = -1
    stop = -1
    if data[0] == 1:
        start = 0
    for i in range(len(data) - 1):
        if data[i] == 0 and data[i + 1] == 1:
            start = i + 1;
        if data[i] == 1 and data[i + 1] == 0:
            stop = i + 1;
            segments.append(Segment(start, stop));
    if data[-1] == 1:
        stop = len(data)
        segments.append(Segment(start, stop));
    return segments

def split(user):
#if __name__ == "__main__":
#    argc = len(sys.argv)
#    if argc != 5:
#        print("Incorrect args. Example:")
#        print("python3 split_by_vad.py dataset/wav/1_2_3_4_5.wav 0.1 0.01 dataset/splitted/")
#        exit(1)

#    wav_file_path = sys.argv[1]
    wav_file_path = "dataset/request/wav/0_0_0_0_0.wav"
#    segment_duration = float(sys.argv[2])
#    vad_threshold = float(sys.argv[3])
#    output_wav_directory = sys.argv[4]
    segment_duration = 0.1
    vad_threshold = 0.01
    output_wav_directory = "dataset/request/splitted/"

    print(wav_file_path)
    sample_rate, audio = read(wav_file_path)

    segment_duration_samples = sec2samples(segment_duration, sample_rate)
    segments_energy = get_segments_energy(audio, segment_duration_samples)
    vad_mask = get_vad_mask(segments_energy, vad_threshold)
    segments = mask_compress(vad_mask)

    print_with_timeline(segments_energy, segment_duration, "sec", 10)
    print_with_timeline(vad_mask, segment_duration, "sec", 10)
    print_segments(segments, segment_duration, "sec")

    fname = wav_file_path[-13:-4]
    digits = fname.split("_")
    print(digits)
    if len(digits) == len(segments):
        bot.send_message(user, f"Вы записали плохое аудио и я ничего не понял :(\nНе злитесь, если я не отгадал\nПопробуйте ещё раз")
#    assert len(digits) == len(segments), "Bad threshold"
#    else:
    max_duration = 0
    for segment in segments:
        duration = (segment.stop - segment.start) * segment_duration_samples / sample_rate
        if duration > max_duration:
            max_duration = duration
    print(max_duration)
    if max_duration <= 0.6:
        bot.send_message(user, f"Я просил назвать 5 цифр с паузами, а вы сделали что-то другое :(\nНе злитесь, если я не отгадал\nПопробуйте ещё раз")
#    assert max_duration <= 0.6, f"max_duration={max_duration:.3f}"

    position = 0
    for digit, segment in zip(digits, segments):
        new_wav_file_path = f"{output_wav_directory}/{position}.wav"

        start = segment.start * segment_duration_samples
        stop = segment.stop * segment_duration_samples
        print(new_wav_file_path, start, stop)
        write(new_wav_file_path, sample_rate, audio[start:stop])

        position += 1
#END OF REQUEST SPLITTING
    answers = []
    req_audios = []
    reqs_path = root + "/request/splitted"
    for file in sorted(os.listdir(reqs_path)):
        file_path = reqs_path + "/" + file
        req_sample_rate, req_audio = read(file_path)
        req_audios.append(req_audio)
    max_duration = int(max_duration_sec * req_sample_rate + 1e-6)
# added part of bot |||| here i try to transform audio and predict num
    reqs = []
    reqs_flatten = []
    for req_audio in req_audios:
        if len(req_audio) < max_duration:
            req_audio = np.pad(req_audio, (0, max_duration - len(req_audio)), mode='constant')
#    elif len(req_audio) > max_duration:
  #      print("LONG AUDIO")
   #     exit(1)

#INSERTED START

#INSERTED end
        request = librosa.feature.melspectrogram(req_audio.astype(float), 
              req_sample_rate, n_mels=16, fmax=1000)
        reqs.append(request)
       # print(reqs.shape)
        reqs_arr = np.array(reqs)
        reqs_flatten.append(request.reshape(-1))
    d2_reqs_arr = reqs_arr.reshape((5, 16*57))
#transformation end
    answers = model.predict(d2_reqs_arr)
    print(answers)
    bot.send_message(user, str(answers))
    bot.send_message(user, "Надеюсь, я угадал")

if __name__ == "__main__":
    bot.polling(none_stop=True, interval=0)
