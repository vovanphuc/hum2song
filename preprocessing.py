import argparse
import yaml
import os

import librosa
import numpy as np
from tqdm import tqdm
import random
from shutil import copyfile

import audio as Audio
import joblib

try:
    from utils.clean_data import clean_train_set, clean_test_set
    from utils.copy_files import copy_fail_cleaning_data
except ModuleNotFoundError:
    from clean_data import clean_train_set, clean_test_set
    from copy_files import copy_fail_cleaning_data

import warnings
warnings.filterwarnings('ignore')

def process(audio, max_wav_value, STFT):
    audio = audio.astype(np.float32)
    audio = audio / max(abs(audio)) * max_wav_value
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(audio, STFT)
    return mel_spectrogram.T

def process_file(audio_path, out_path, sampling_rate, max_wav_value, STFT):
    audio, _ = librosa.load(audio_path, sr=sampling_rate)
    spec = process(audio, max_wav_value, STFT)
    np.save(out_path, spec)

def process_data(dataset, in_dir, out_dir, sampling_rate, max_wav_value, STFT, val_size=0):
    random.seed(1234)

    subfolder = ["hum", "song", "full_song"]
    print(f"Processing {dataset}")
    for sub in subfolder:
        if not os.path.isdir(os.path.join(in_dir, dataset, sub)):
            continue
        files = os.listdir(os.path.join(in_dir, dataset, sub))
        os.makedirs(os.path.join(out_dir, dataset, sub), exist_ok=True)

        ## complementary: Multi-processing
        in_list = []
        out_list = []
        for file in files:
            if file[-4:] != ".mp3":
                continue
            # preprocess audio
            audio_path = os.path.join(in_dir, dataset, sub, file)
            out_path = os.path.join(out_dir, dataset, sub, f"{file[:-4]}.npy")
            in_list.append(audio_path)
            out_list.append(out_path)

        jobs = [ joblib.delayed(process_file)(i, o, sampling_rate, max_wav_value, STFT) 
                                                for i,o in zip(in_list, out_list) ]
        joblib.Parallel(n_jobs=4, verbose=1)(jobs)
        ##

    # if dataset == "train":
    #     data_list = os.listdir(os.path.join(out_dir, dataset, "hum"))
    #     data_list = [data for data in data_list]
    #     random.shuffle(data_list)
    #     val_data = data_list[-val_size:]
    #     subfolder = ["hum", "song"]
    #     print(f"Preparing val set")
    #     for sub in subfolder:
    #         os.makedirs(os.path.join(out_dir, "val", sub), exist_ok=True)
    #         for file in tqdm(val_data):
    #             if file[-4:] != ".npy":
    #                 continue
    #             src = os.path.join(out_dir, "train", sub, file)
    #             dst = os.path.join(out_dir, "val", sub, file)
    #             os.rename(src, dst)

def main(config):
    in_dir = config["path"]["raw_path"]
    temp_dir = config["path"]["temp_dir"]
    out_dir = config["path"]["preprocessed_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    val_size = config["preprocessing"]["val_size"]

    STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    clean_train_set(in_dir, out_dir=temp_dir)
    clean_test_set(in_dir, out_dir=temp_dir)
    copy_fail_cleaning_data(in_dir, temp_dir)
    process_data("train", temp_dir, out_dir, sampling_rate, max_wav_value, STFT, val_size)
    process_data("public_test", temp_dir, out_dir, sampling_rate, max_wav_value, STFT)
    copyfile(os.path.join(in_dir, "train", "train_meta.csv"),
              os.path.join(out_dir, "train_meta.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, 
                        default="config/preprocess.yaml",
                        help="path to preprocess.yaml")
    parser.add_argument("--indir", type=str, required=False, help="path to input")
    parser.add_argument("--tempdir", type=str, required=False, help="path to input")
    parser.add_argument("--outdir", type=str, required=False, help="path to output")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if args.indir is not None:
        config["path"]["raw_path"] = args.indir
    if args.tempdir is not None:
        config["path"]["temp_dir"] = args.tempdir
    if args.outdir is not None:
        config["path"]["preprocessed_path"] = args.outdir
    main(config)
