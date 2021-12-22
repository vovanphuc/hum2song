import augment
import numpy as np
import torch
import torchaudio
import argparse
import yaml
import os
import random
from tqdm import tqdm
import threading

def aug_pitch(audio, sr=16000, low_pitch=-350, high_pitch=300):
    random_pitch_shift = lambda: np.random.randint(low_pitch, high_pitch)
    y = augment.EffectChain().pitch(random_pitch_shift).rate(sr).apply(audio, src_info={'rate': sr})
    return y

def aug_reverb(audio, sr=16000, max_reverb=100):
    y = augment.EffectChain().reverb(random.randint(1, max_reverb), 
                                     random.randint(1, max_reverb), 
                                     random.randint(1, max_reverb)) \
                            .channels(1) \
                            .apply(audio, src_info={'rate': sr})
    return y

def aug_dropout(audio, sr=16000, max_time_drop=0.3, max_times=8):
    effect = augment.EffectChain()
    for _ in range(random.randint(1, max_times)):
        effect = effect.time_dropout(max_seconds=max_time_drop)
    y = effect.apply(audio, src_info={'rate': sr})
    return y

def aug_tempo(audio, sr=16000, tempo=1, min_tempo=0.85, max_tempo=1.15):
    if tempo == 1:
        while abs(tempo - 1) < 0.01:
            tempo = random.uniform(min_tempo, max_tempo)
    y = augment.EffectChain().tempo(tempo).apply(audio, src_info={'rate': sr})
    return y

def aug_noise(audio, sr=16000, low_noise=5, high_noise=13):    
    noise_generator = lambda: torch.zeros_like(audio).uniform_()
    y = augment.EffectChain().additive_noise(noise_generator, snr=random.randint(low_noise, high_noise)) \
                            .apply(audio, src_info={'rate': sr})
    return y

def aug_sinc(audio, sr=16000, min_sinc=40, max_sinc=180):
    sinc = random.randint(min_sinc, max_sinc)
    y = augment.EffectChain().sinc('-a', str(sinc), '500-100').apply(audio, src_info={'rate': sr})
    return y

def aug_gain(audio, sr=16000, volume=25):
    gain_volume = 0
    while gain_volume == 0:
        gain_volume = random.randint(-volume, volume)
    y = augment.EffectChain().gain(gain_volume).apply(audio, src_info={'rate': sr})
    return y

def aug_combination(audio, sr=16000, path=None):
    # effect_list = [aug_pitch, aug_reverb, aug_dropout, aug_tempo, aug_noise, aug_sinc, aug_gain]
    effect_list = [aug_pitch, aug_reverb, aug_dropout, aug_tempo, aug_sinc]
    num_to_select = random.randint(3, len(effect_list))
    effects = random.sample(effect_list, num_to_select)
    for effect in effects:
        audio = effect(audio, sr=sr)

    torchaudio.save(path, audio, sr, format='mp3')

def main(config):
    random.seed(1234)

    temp_dir = config["path"]["temp_dir"]
    subs = ["hum", "song"]

    for sub in subs:
        sound_path = os.path.join(temp_dir, "train", sub)
        tries = config["tries"]

        aug_path = os.path.join(temp_dir, 'augment', 'train', sub)
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