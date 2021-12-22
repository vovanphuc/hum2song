import argparse
import yaml
import os
from tqdm import tqdm
import librosa
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import audio as Audio
import warnings
warnings.filterwarnings('ignore')

def process(audio, max_wav_value, STFT):
    audio = audio.astype(np.float32)
    audio = audio / max(abs(audio)) * max_wav_value
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(audio, STFT)
    return mel_spectrogram.T

def process_augment_data(dataset, in_dir, out_dir, sampling_rate, max_wav_value, STFT):
    subfolder = ["hum", "song", "full_song"]
    print(f"Processing {dataset}")

    os.makedirs(os.path.join(out_dir, "augment"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "augment", dataset), exist_ok=True)
    for sub in subfolder:
        aug_path = os.path.join(in_dir, "augment", dataset, sub)
        if not os.path.isdir(aug_path):
            continue
        files = os.listdir(aug_path)

        os.makedirs(os.path.join(out_dir, "augment", dataset, sub), exist_ok=True)
        for file in tqdm(files):
            if file[-4:] != ".mp3":
                continue
            # preprocess audio
            audio_path = os.path.join(aug_path, file)
            audio, _ = librosa.load(audio_path, sr=sampling_rate)
            spec = process(audio, max_wav_value, STFT)
            np.save(os.path.join(out_dir, "augment", dataset, sub, f"{file[:-4]}.npy"), spec)

def main(config):
    temp_dir = config["path"]["temp_dir"]
    out_dir = config["path"]["preprocessed_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]

    STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    process_augment_data("train", temp_dir, out_dir, sampling_rate, max_wav_value, STFT)
    # process_augment_data("public_test", temp_dir, out_dir, sampling_rate, max_wav_value, STFT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, 
                        default="config/preprocess.yaml",
                        help="path to preprocess.yaml")
    parser.add_argument("--tempdir", type=str, required=False, help="path to input")
    parser.add_argument("--outdir", type=str, required=False, help="path to output")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if args.tempdir is not None:
        config["path"]["temp_dir"] = args.tempdir
    if args.outdir is not None:
        config["path"]["preprocessed_path"] = args.outdir
    main(config)
