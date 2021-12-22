import os
import numpy as np
import argparse
import yaml
from tqdm import tqdm

def main(config):
    indir = config["path"]["preprocessed_path"]
    win_size = config["preprocessing"]["split_win_size"]
    stride_size = config["preprocessing"]["split_stride_size"]

    full_song_path = os.path.join(indir, "public_test/full_song")
    assert os.path.isdir(full_song_path), f"Not found full song folder at {full_song_path}"
    assert len(os.listdir(full_song_path)), f"Full song folder is empty at {full_song_path}"

    song_path = os.path.join(indir, "public_test/song")
    os.makedirs(song_path)
    for fsong in tqdm(os.listdir(full_song_path)):
        if fsong[-4:] != ".npy":
            continue
        fsong_id = fsong[:-4]

        fsong_np = np.load(os.path.join(full_song_path, fsong))
        start = 0
        song_id = 0
        while True:
            if fsong_np[start:, :].shape[0] < win_size: # remain is shortter than win_size
                if fsong_np[start:, :].shape[0] < win_size - stride_size // 2: # remain is shortter than win_size 1s
                    break
                song = fsong_np[-win_size:, :]
            else:
                song = fsong_np[start: start + win_size, :]
            song_name = f"{fsong_id}_{song_id}.npy"
            np.save(os.path.join(song_path, song_name), song)

            start += stride_size
            song_id += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, 
                        default="config/preprocess.yaml",
                        help="path to preprocess.yaml")
    parser.add_argument("--indir", type=str, required=False, help="path to input/outdir")
    parser.add_argument("--win_size", type=int, required=False, help="len of splited mel")
    parser.add_argument("--stride_size", type=int, required=False, help="stride splited mel")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if args.indir is not None:
        config["path"]["preprocessed_path"] = args.indir
    if args.win_size is not None:
        config["preprocessing"]["split_win_size"] = args.win_size
    if args.stride_size is not None:
        config["preprocessing"]["split_stride_size"] = args.stride_size

    main(config)