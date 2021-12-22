import numpy as np
import yaml
import argparse
import os
import random
import matplotlib.pyplot as plt

def save_img(path, spec_song=None, spec_hum=None):
    if spec_song is None or spec_hum is None:
        if spec_song is not None:
            plt.imshow(spec_song, origin="lower")
            plt.title("song", fontsize="medium")
            plt.ylim(0, spec_song.shape[0])
        if spec_hum is not None:
            plt.imshow(spec_hum, origin="lower")
            plt.title("hum", fontsize="medium")
            plt.ylim(0, spec_hum.shape[0])
    else:
        fig, axes = plt.subplots(2, 1, squeeze=False)
        axes[0, 0].imshow(spec_song, origin="lower")
        axes[0, 0].set_title("song", fontsize="medium")
        axes[0, 0].set_ylim(0, spec_song.shape[0])
        axes[1, 0].imshow(spec_hum, origin="lower")
        axes[1, 0].set_title("hum", fontsize="medium")
        axes[1, 0].set_ylim(0, spec_hum.shape[0])

    plt.savefig(path)
    plt.close()

def visualize(dataset, in_dir, out_dir, num):
    random.seed(1234)

    files = os.listdir(os.path.join(in_dir, dataset, "hum"))
    random.shuffle(files)
    files = random.sample(files, k=min(num, len(files)))

    os.makedirs(os.path.join(out_dir, dataset), exist_ok=True)
    if dataset == "train" or dataset == "val":
        for file in files:
            spec_hum = np.load(os.path.join(in_dir, dataset, "hum", file))
            spec_song = np.load(os.path.join(in_dir, dataset, "song", file)) 
            save_img(os.path.join(out_dir, dataset, file[:-4] + ".jpg"), spec_song.T, spec_hum.T)
    elif dataset == "public_test":
        os.makedirs(os.path.join(out_dir, dataset, "hum"), exist_ok=True)
        for file in files:
            spec_hum = np.load(os.path.join(in_dir, dataset, "hum", file)) 
            save_img(os.path.join(out_dir, dataset, "hum", file[:-4] + ".jpg"), spec_hum=spec_hum.T)

        files = os.listdir(os.path.join(in_dir, dataset, "full_song"))
        random.shuffle(files)
        files = random.sample(files, k=min(num, len(files)))
        
        os.makedirs(os.path.join(out_dir, dataset, "full_song"), exist_ok=True)
        for file in files:
            spec_song = np.load(os.path.join(in_dir, dataset, "full_song", file)) 
            save_img(os.path.join(out_dir, dataset, "full_song", file[:-4] + ".jpg"), spec_song=spec_song.T)

def main(config, num):
    in_dir = config["path"]["preprocessed_path"]
    out_dir = config["path"]["visualization_path"]
    dataset = ["train", "val", "public_test"]
    os.makedirs(out_dir, exist_ok=True)
    for data in dataset:
        visualize(data, in_dir, out_dir, num)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, 
                        default="config/preprocess.yaml",
                        help="path to preprocess.yaml")
    parser.add_argument("--indir", type=str, required=False, help="path to input")
    parser.add_argument("--outdir", type=str, required=False, help="path to output")
    parser.add_argument("--num", type=int, required=False, default=5, help="num of samples")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if args.indir is not None:
        config["path"]["preprocessed_path"] = args.indir
    if args.outdir is not None:
        config["path"]["visualization_path"] = args.outdir

    main(config, num=args.num)