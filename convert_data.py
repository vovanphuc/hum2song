import os
import random
import csv
from config.config import Config

config = Config()
path_root = config.train_root
csv_path = config.meta_train
path_train_hum = os.path.join(path_root, 'train', 'hum')
# path_train_song = os.path.join(path_root, 'train', 'song')

path_val_hum = os.path.join(path_root, 'val', 'hum')
# path_val_song = os.path.join(path_root, 'val', 'song')

path_aug_hum = os.path.join(path_root, 'augment')

list_train_hum = sorted(os.listdir(path_train_hum))
list_val_hum = sorted(os.listdir(path_val_hum))

def read_meta_data(csv_path):
    data_info = []
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                music_id = row[0]
                song_path = row[1]
                hum_path = row[2]
                data_info.append([music_id, song_path, hum_path])
            line_count += 1
        print(f'Processed {line_count} lines.')
    return data_info
    
def create_meta_dict(train_meta):
    sound2id = {}
    id2hum = {}
    id2song = {}
    id2stt = {}

    stt = 0
    for row in train_meta:
        music_id = row[0]
        song_name = row[1].split("/")[-1].split(".")[0]
        hum_name = row[2].split("/")[-1].split(".")[0]
        sound2id[song_name] = music_id
        sound2id[hum_name] = music_id

        if music_id not in id2hum.keys():
            id2hum[music_id] = [hum_name]
            id2song[music_id] = [song_name]
        else:
            id2hum[music_id].append(hum_name)
            id2song[music_id].append(song_name)

        if music_id not in id2stt.keys():
            id2stt[music_id] = stt
            stt += 1
    return sound2id, id2hum, id2song, id2stt

def writeFile(fileName, content):
    with open(fileName, 'a') as f1:
        f1.write(content + os.linesep)

def convert_data(id2stt, sound2id, path_txt, list_train_hum=None, list_val_hum=None, path_aug_hum=None):
    if os.path.isfile(path_txt):
        os.remove(path_txt)

    check_song = {}
    if list_train_hum is not None:
        for name_hum in list_train_hum:
            if id2stt == None and "aug" in name_hum:
                continue
            if "aug" in name_hum:
                hum_id = name_hum.split("_")[0]
            else:
                hum_id = name_hum.split(".")[0]
            if id2stt == None:
                label = sound2id[hum_id]
            else:
                label = id2stt[sound2id[hum_id]]
            writeFile(path_txt, f'train/hum/{name_hum} {label}')
            if sound2id[hum_id] not in check_song.keys():
                check_song[sound2id[hum_id]] = True
                writeFile(path_txt, f'train/song/{name_hum} {label}')
    if list_val_hum is not None:
        for name_hum in list_val_hum:
            if id2stt == None and "aug" in name_hum:
                continue
            if "aug" in name_hum:
                hum_id = name_hum.split("_")[0]
            else:
                hum_id = name_hum.split(".")[0]
            if id2stt == None:
                label = sound2id[hum_id]
            else:
                label = id2stt[sound2id[hum_id]]
            writeFile(path_txt, f'val/hum/{name_hum} {label}')
            if sound2id[hum_id] not in check_song.keys():
                check_song[sound2id[hum_id]] = True
                writeFile(path_txt, f'val/song/{name_hum} {label}')
    if path_aug_hum is not None:
        dataset = [data for data in os.listdir(path_aug_hum) 
                        if os.path.isdir(os.path.join(path_aug_hum, data))]

        for data in dataset:
            if "test" in data:
                continue

            if "test" in data:
                list_aug = sorted(os.listdir(os.path.join(path_aug_hum, data, "full_song")))
            else:
                list_aug = sorted(os.listdir(os.path.join(path_aug_hum, data, "hum")))
            
            for name_hum in list_aug:
                hum_id = name_hum.split("_")[0]
                if sound2id[hum_id] not in check_song.keys():
                    continue
                
                label = id2stt[sound2id[hum_id]]
                if os.path.isfile(os.path.join(path_aug_hum, f'{data}/hum/{name_hum}')):
                    writeFile(path_txt, f'augment/{data}/hum/{name_hum} {label}')
                if os.path.isfile(os.path.join(path_aug_hum, f'{data}/full_song/{name_hum}')):
                    writeFile(path_txt, f'augment/{data}/full_song/{name_hum} {label}')
    print(f"Saving {path_txt}")

train_meta = read_meta_data(csv_path)
sound2id, _, _, id2stt = create_meta_dict(train_meta)

convert_data(id2stt, sound2id, 'data_train.txt', list_train_hum, list_val_hum=None, path_aug_hum=path_aug_hum)
convert_data(id2stt, sound2id, 'full_data_train.txt', list_train_hum, list_val_hum, path_aug_hum=path_aug_hum)

convert_data(id2stt=None, sound2id=sound2id, path_txt='data_val.txt', list_train_hum=None, list_val_hum=list_val_hum)
convert_data(id2stt=None, sound2id=sound2id, path_txt='full_data_val.txt', list_train_hum=list_train_hum, list_val_hum=list_val_hum)