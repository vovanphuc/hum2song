import os, json, faiss
from utils.utils import *
from tqdm import tqdm
from models.resnet import *
from config.config import Config

def get_vector2index():
    return faiss.IndexFlatL2(512)


class CFG():
    def __init__(self):
        self.vector2index = get_vector2index()


def read_val(path_val, data_root):
    files = open(path_val, 'r+')
    dict_data = []
    for line in files.read().splitlines():
        if 'song' in line:
            type = 'song'
        else:
            type = 'hum'
        dict_data.append({
            'path': os.path.join(data_root, line.split(' ')[0]),
            'label': line.split(' ')[1],
            'type': type
        })
    files.close()
    return dict_data

def search_vector(path_hum, model, index2id, cfg, input_shape):
    image = load_image(path_hum, input_shape)
    feature = get_feature(model, image)
    distances, lst_index = cfg.vector2index.search(feature, k=10)
    lst_result = []
    for index in lst_index[0]:
        result = str(index2id[str(index)])
        lst_result.append(result)

    return lst_result, distances

def mean_reciprocal_rank(preds, gt: str, k: int=10):
    preds = preds[: min(k, len(preds))]
    score = 0
    for rank, pred in enumerate(preds):
        if pred == gt:
            score = 1 / (rank + 1)
            break
    return score

def mrr_score(model, dict_data, input_shape):
    index2id = {"-1": ""}
    id = 0
    cfg = CFG()
    count = 0
    s_0 = 0
    s_1 = 0
    result_search = []
    for item in tqdm(dict_data):
        if item['type'] == 'song':
            path_song = item['path']
            image = load_image(path_song, input_shape)
            cfg.vector2index.add(get_feature(model, image))
            index2id[str(id)] = item['label']
            id += 1
    for item in tqdm(dict_data):
        if item['type'] == 'hum':
            path_hum = item['path']
            preds, distances = search_vector(path_hum, model, index2id, cfg, input_shape)
            result_search.append([item['label'], preds])

            mrr = mean_reciprocal_rank(preds, item['label'])
            if mrr == 1.0:
                s_0 += distances[0, 0]
                s_1 += distances[0, 1]
                count += 1
    mrr = 0
    for row in result_search:
        mrr += mean_reciprocal_rank(row[1], row[0])
    mrr = mrr/len(result_search)

    print("AVG", s_0 / count, s_1 / count, count)
    return mrr


if __name__ == '__main__':
    config = Config()

    model = wrap_resnet_face18(config.use_se)
    model.load_state_dict(torch.load(os.path.join(config.checkpoints_path, 'resnet18_latest.pth')))
    model.to('cuda')
    model.eval()
    dict_data = read_val(config.val_list, config.train_root)
    mrr = mrr_score(model, dict_data, config.input_shape)
    print("-----------------------------")
    print(f"MRR: {mrr}")