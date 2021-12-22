import torch
import numpy as np
from torchvision import transforms as T
from sklearn.preprocessing import normalize
import os


def load_image(npy_path, input_shape=(630, 80)):
    data = np.load(npy_path)
    if data.shape[0] >= input_shape[0]:
        result = data[:input_shape[0], :]
    else:
        result = np.zeros(input_shape)
        result[:data.shape[0], :data.shape[1]] = data
    image = torch.from_numpy(result).unsqueeze(0).unsqueeze(0)
    return image.float()


def get_feature(model, image):
    data = image.to(torch.device("cuda"))
    with torch.no_grad():
        output = model(data)
    output = output.cpu().detach().numpy()
    output = normalize(output).flatten()
    return np.matrix(output)


def writeFile(fileName, content):
    with open(fileName, 'a') as f1:
        f1.write(content + os.linesep)
