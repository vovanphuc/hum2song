import argparse
import os, json, faiss
from models.resnet import *
from utils.utils import *
from tqdm import tqdm
from config.config import Config
import time
import argparse

config = Config()
start_time_load_model = time.time()
model = wrap_resnet_face18(False)
model.load_state_dict(torch.load(os.path.join(config.checkpoints_path, 'resnet18_latest.pth')))
model.to('cuda')
model.eval()
print('TIME LOAD MODEL: ', time.time() - start_time_load_model)


def get_json_dict(path):
    if os.path.exists(path):
        with open(path, mode='r', encoding='utf-8') as _f:
            return json.load(_f)
    else:
        return {}


def get_vector2index():
    return faiss.IndexFlatL2(512)


class CFG():
    def __init__(self):
        self.vector2index = get_vector2index()


def search_vector(path_hum, cfg, index2id, input_shape):
    image = load_image(path_hum, input_shape)
    feature = get_feature(model, image)
    _, lst_index = cfg.vector2index.search(feature, k=30)
    lst_result = []
    for index in lst_index[0]:
        result = str(index2id[str(index)]).split('_')[0]
        if result not in lst_result:
            lst_result.append(result)
        if len(lst_result) == 10:
            break
    _result = ''
    for index in lst_result[:10]:
        _result += f",{index}"
    return _result


def create_submit(root_song, root_hum, path_result, input_shape):
    try:
        os.remove(path_result)
    except:
        pass
    cfg = CFG()
    list_song = os.listdir(root_song)
    index2id = {"-1": ""}
    for id, name_song in tqdm(enumerate(list_song)):
        path_song = os.path.join(root_song, name_song)
        image = load_image(path_song, input_shape)
        cfg.vector2index.add(get_feature(model, image))
        index2id[str(id)] = name_song.split('.')[0]

    lst_hum = sorted(os.listdir(root_hum))
    for _, name_hum in tqdm(enumerate(lst_hum)):
        path_hum = os.path.join(root_hum, name_hum)
        rsult_song = search_vector(path_hum, cfg, index2id, input_shape)
        writeFile(path_result, f'{name_hum.replace("npy", "mp3")}{rsult_song}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/preprocessed/public_test", required=False, help="path to data")
    parser.add_argument("--output", type=str, default="/result/submission.csv", required=False, help="path to output")
    args = parser.parse_args()

    config = Config()
    parser = argparse.ArgumentParser()
    start_time_infer = time.time()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    is_song = 'song' if os.path.isdir(os.path.join(args.data, 'song')) else 'full_song'
    create_submit(os.path.join(args.data, is_song),
                  os.path.join(args.data, 'hum'),
                  args.output,
                  config.input_shape)
    print('TIME INFERENCE :', time.time() - start_time_infer)
