import argparse
import yaml
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import audio as Audio
from clean_data import clean_test_set
from copy_files import copy_fail_cleaning_data
from preprocessing import process_data


def main(config, test_type):
    in_dir = config["path"]["raw_path"]
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

    print("Preprocessing...")
    clean_test_set(in_dir, out_dir=temp_dir, test_type=test_type)
    copy_fail_cleaning_data(in_dir, temp_dir, test_type)
    process_data(test_type, temp_dir, out_dir, sampling_rate, max_wav_value, STFT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, 
                        default="config/preprocess.yaml",
                        help="path to preprocess.yaml")
    parser.add_argument("--indir", type=str, required=False, help="path to input")
    parser.add_argument("--tempdir", type=str, required=False, help="path to input")
    parser.add_argument("--outdir", type=str, required=False, help="path to output")
    parser.add_argument("--test_type", type=str, required=True, help="public_test/private_test")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if args.indir is not None:
        config["path"]["raw_path"] = args.indir
    if args.tempdir is not None:
        config["path"]["temp_dir"] = args.tempdir
    if args.outdir is not None:
        config["path"]["preprocessed_path"] = args.outdir
    main(config, args.test_type)
