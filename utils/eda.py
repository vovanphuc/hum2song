import yaml
import argparse
import os
import csv
import sys
from tqdm import tqdm
import logging

sys.path.append(os.path.dirname(__file__))
from get_valid_interval import get_valid_interval
from pydub import AudioSegment

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

def write_meta_data(header, data, csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write multiple rows
        writer.writerows(data)

def add_valid_interval_field(data_info, raw_path):
    for idx in tqdm(range(len(data_info))):
        row = data_info[idx]
        song_path = os.path.join(raw_path, row[1])
        hum_path = os.path.join(raw_path, row[2])
        if not os.path.isfile(song_path) or not os.path.isfile(hum_path):
            if not os.path.isfile(song_path):
                logging.warn(f"Not found {song_path}")
            if not os.path.isfile(hum_path):
                logging.warn(f"Not found {hum_path}")
            continue
    
        song = AudioSegment.from_file(song_path, format="mp3")
        hum = AudioSegment.from_file(hum_path, format="mp3")
        song_org_dur = len(song) / 1000
        hum_org_dur = len(hum) / 1000
        song_interval = get_valid_interval(song)
        hum_interval = get_valid_interval(hum)
        data_info[idx] += [song_org_dur,
                              hum_org_dur,
                              (song_interval[1] - song_interval[0])/1000, 
                              (hum_interval[1] - hum_interval[0])/1000]
        # print(idx, data_info[idx])
    return data_info

def add_valid_interval_field_for_test(data_info, raw_path):
    for idx in range(len(data_info)):
        row = data_info[idx]
        sound_path = os.path.join(raw_path, row[0])
        sound = AudioSegment.from_file(sound_path, format="mp3")
        sound_org_dur = len(sound) / 1000
        sound_interval = get_valid_interval(sound)
        data_info[idx] += [sound_org_dur,
                              (sound_interval[1] - sound_interval[0])/1000]
        print(idx, data_info[idx])
    return data_info

def create_meta_data_test(test_path):
    song_info = []
    hum_info = []
    for file in os.listdir(os.path.join(test_path, "full_song")):
        song_path = os.path.join("public_test", "full_song", file)
        song_info.append([song_path])
    for file in os.listdir(os.path.join(test_path, "hum")):
        hum_path = os.path.join("public_test", "hum", file)
        hum_info.append([hum_path])

    return song_info, hum_info
def main(config):
    raw_path = config["path"]["raw_path"]
    indir = config["path"]["preprocessed_path"]

    # eda train
    data_info = read_meta_data(os.path.join(raw_path, "train", "train_meta.csv"))
    data_info = add_valid_interval_field(data_info, raw_path)

    header = ["music_id", "song_path", "hum_path", 
            "song_org_dur", "hum_org_dur", 
            "song_valid_interval", "hum_valid_interval"]
    write_meta_data(header, data_info, os.path.join(indir, "train", "eda_train_meta.csv"))

    # eda test
    song_info, hum_info = create_meta_data_test(os.path.join(raw_path, "public_test"))
    song_info = add_valid_interval_field_for_test(song_info, raw_path)
    header = ["song_path", "song_org_dur", "song_valid_interval"]
    write_meta_data(header, song_info, os.path.join(indir, "public_test", "eda_song_meta.csv"))

    hum_info = add_valid_interval_field_for_test(hum_info, raw_path)
    header = ["hum_path", "hum_org_dur", "hum_valid_interval"]
    write_meta_data(header, hum_info, os.path.join(indir, "public_test", "eda_hum_meta.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, 
                        default="config/preprocess.yaml",
                        help="path to preprocess.yaml")
    parser.add_argument("--indir", type=str, required=False, help="path to input")
    parser.add_argument("--outdir", type=str, required=False, help="path to output")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if args.indir is not None:
        config["path"]["preprocessed_path"] = args.indir
    if args.outdir is not None:
        config["path"]["visualization_path"] = args.outdir

    main(config)