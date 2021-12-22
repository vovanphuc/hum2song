import argparse
import csv
import sys
import os

from typing import List

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

def mean_reciprocal_rank(preds: List[str], gt: str, k: int=10):
    preds = preds[: min(k, len(preds))]
    score = 0
    for rank, pred in enumerate(preds):
        if pred == gt:
            score = 1 / (rank + 1)
            break
    return score

def main(csv_pred, csv_gt):
    train_meta = read_meta_data(csv_gt)
    sound2id, _, _, _ = create_meta_dict(train_meta)
    
    acc = 0
    num = 0
    with open(csv_pred, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                hum_id = row[0].split(".")[0]
                preds = []
                for col in row[1:]:
                    preds.append(sound2id[str(col)])

                print(hum_id, mean_reciprocal_rank(preds, sound2id[str(hum_id)]))
                acc += mean_reciprocal_rank(preds, sound2id[str(hum_id)])
                num += 1
            line_count += 1
        print(f'Processed {line_count} lines.')
    return acc / num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_pred", type=str, required=True, help="path to predict csv")
    parser.add_argument("--csv_gt", type=str, required=True, help="path to ground-truth csv")
    args = parser.parse_args()

    mrr = main(args.csv_pred, args.csv_gt)
    print("-----------------------------")
    print(f"MRR: {mrr}")