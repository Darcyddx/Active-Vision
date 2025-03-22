from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from glob import glob
import pandas as pd
import shutil

HEIGHT = 288
WIDTH = 512
mag = 1
sigma = 2.5

def genHeatMap(w, h, cx, cy, r, mag):
    if cx < 0 or cy < 0:
        return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1)) ** 2) + ((x - (cx + 1)) ** 2)
    heatmap[heatmap <= r ** 2] = 1
    heatmap[heatmap > r ** 2] = 0
    return heatmap * mag

root_path = "/root/autodl-tmp/dataset/tennis-data-v2"
game_list = ['game1', 'game2', 'game3', 'game4', 'game5', 'game6', 'game7', 'game8', 'game9']
p = os.path.join(root_path, game_list[0], 'frame', 'Clip1', '0000.jpg')
a = img_to_array(load_img(p))  # Load as grayscale
ratio = a.shape[0] / HEIGHT

dataDir = "/root/autodl-tmp/dataset/npz-rgb"
if os.path.exists(dataDir):
    shutil.rmtree(dataDir)
os.makedirs(dataDir)

count = 1

for game in game_list:
    all_path = glob(os.path.join(root_path, game, 'frame', '*'))
    train_path = all_path[:]
    for i in range(len(train_path)):
        train_path[i] = train_path[i][len(os.path.join(root_path, game, 'frame')) + 1:]
    for p in train_path:
        print(game, p)
        labelPath = os.path.join(root_path, game, 'ball_trajectory', p + '_ball.csv')
        data = pd.read_csv(labelPath)
        no = data['file name'].values
        v = data['visibility'].values
        x = data['x-coordinate'].values
        y = data['y-coordinate'].values
        num = no.shape[0]
        r = os.path.join(root_path, game, 'frame', p)
        x_data_tmp = []
        y_data_tmp = []
        for i in range(num - 2):
            unit = []
            for j in range(3):
                target = str(no[i + j])
                png_path = os.path.join(root_path, r, target)
                a = load_img(png_path)
                a = img_to_array(a.resize(size=(WIDTH, HEIGHT)))
                unit.append(a[:, :, 0])
                unit.append(a[:, :, 1])
                unit.append(a[:, :, 2])
                del a
            x_data_tmp.append(np.stack(unit, axis=-1))  # Combine 3 frames as channels
            del unit
            unit = []
            for j in range(3):
                if v[i + j] == 0:
                    unit.append(genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag))
                else:
                    unit.append(genHeatMap(WIDTH, HEIGHT, int(x[i + j] / ratio), int(y[i + j] / ratio), sigma, mag))
            y_data_tmp.append(np.stack(unit, axis=-1))  # Combine 3 heatmaps as channels
            del unit

        x_data = np.asarray(x_data_tmp, dtype="float32") / 255.0
        y_data = np.asarray(y_data_tmp, dtype="float32")

        np.savez_compressed(os.path.join(dataDir, f'data_{count}.npz'), x_data=x_data, y_data=y_data)

        print('============================')
        print(count)
        print(game, p)
        print(x_data.shape)
        print(y_data.shape)
        print('============================')
        del x_data
        del y_data
        count += 1
