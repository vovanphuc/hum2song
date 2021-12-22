import numpy as np
import argparse
import yaml
import os
import random
from tqdm import tqdm
from augment_mp3 import *
import torchaudio

def main(config):
    random.seed(1234)

    temp_dir = config["path"]["temp_dir"]
    subs = ["full_song"]

    for sub in subs:
        sound_path = os.path.join(temp_dir, "public_test", sub)
        tries = config["tries"]

        aug_path = os.path.join(temp_dir, 'augment', 'public_test', sub)
        os.makedirs(aug_path, exist_ok=True)

        thds = []
        for file in tqdm(os.listdir(sound_path)):
            audio, sr = torchaudio.load(os.path.join(sound_path, file))

            for i in range(tries):
                filename = file[:-4] + "_aug" + str(i) + file[-4:]
                t1 = threading.Thread(target=aug_combination, args=(audio, sr, os.path.join(aug_path, filename),))
                thds.append(t1)
                t1.start()
        for t in thds:
            t.join()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, 
                        default="config/preprocess.yaml",
                        help="path to preprocess.yaml")
    parser.add_argument("--tempdir", type=str, required=False, help="path to input/outdir")
    parser.add_argument("--tries", type=int, default=5, required=False, help="number of tries")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config["tries"] = args.tries
    if args.tempdir is not None:
        config["path"]["temp_dir"] = args.tempdir

    main(config)