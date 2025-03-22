import numpy as np
import sys, getopt
import os
from glob import glob
import tensorflow as tf
import cv2
from os.path import isfile, join
from PIL import Image
import time

BATCH_SIZE = 1
HEIGHT = 288
WIDTH = 512
sigma = 2.5
mag = 1


def genHeatMap(w, h, cx, cy, r, mag):
    if cx < 0 or cy < 0:
        return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1)) ** 2) + ((x - (cx + 1)) ** 2)
    heatmap[heatmap <= r ** 2] = 1
    heatmap[heatmap > r ** 2] = 0
    return heatmap * mag


# Custom time formatting
def custom_time(time):
    remain = int(time / 1000)
    ms = (time / 1000) - remain
    s = remain % 60
    s += ms
    remain = int(remain / 60)
    m = remain % 60
    remain = int(remain / 60)
    h = remain
    return f"{h:02}:{m:02}:{int(s):02}.{int(ms * 1000):03}"


try:
    (opts, args) = getopt.getopt(sys.argv[1:], '', [
        'video_name=',
        'load_weights='
    ])
    if len(opts) != 2:
        raise ''
except:
    print('usage: python3 test.py --video_name=<videoPath> --load_weights=<weightPath>')
    exit(1)

for (opt, arg) in opts:
    if opt == '--video_name':
        videoName = arg
    elif opt == '--load_weights':
        load_weights = arg
    else:
        print('usage: python3 test.py --video_name=<videoPath> --load_weights=<weightPath>')
        exit(1)


# Loss function
def custom_loss(y_true, y_pred):
    loss = (-1) * (K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (
                1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
    return K.mean(loss)


model = tf.keras.models.load_model(load_weights)

model.summary()

start = time.time()

f = open(videoName[:-4] + '_predict.csv', 'w')
f.write('Frame,Visibility,X,Y,Time\n')

cap = cv2.VideoCapture(videoName)

success, image1 = cap.read()
frame_time1 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
success, image2 = cap.read()
frame_time2 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
success, image3 = cap.read()
frame_time3 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))

ratio = image1.shape[0] / HEIGHT
size = (int(WIDTH * ratio), int(HEIGHT * ratio))
fps = 30

if videoName[-3:] == 'avi':
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
elif videoName[-3:] == 'mp4':
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
    print('usage: video type can only be .avi or .mp4')
    exit(1)

out = cv2.VideoWriter(videoName[:-4] + '_predict' + videoName[-4:], fourcc, fps, size)

count = 0

while success:
    x1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    x2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    x3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

    # Resize images to (WIDTH, HEIGHT)
    x1 = cv2.resize(x1, (WIDTH, HEIGHT))
    x2 = cv2.resize(x2, (WIDTH, HEIGHT))
    x3 = cv2.resize(x3, (WIDTH, HEIGHT))

    # Stack images along the channel axis
    # Each image has shape (HEIGHT, WIDTH, 1)
    # Stacked unit will have shape (HEIGHT, WIDTH, 3)
    # unit = np.concatenate((x1, x2, x3), axis=1)
    unit = np.stack((x1, x2, x3), axis=-1)

    # Add a new axis to create a batch dimension
    # Now unit has shape (1, HEIGHT, WIDTH, 9)
    unit = unit[np.newaxis, ...]

    # Convert to float32 and normalize
    unit = unit.astype('float32') / 255.0
   
    # Predict
    y_pred = model.predict(unit, batch_size=BATCH_SIZE)
    y_pred = (y_pred > 0.5).astype("float32")
    y_pred = np.transpose(y_pred, (0, 3, 1, 2))
    h_pred = y_pred[0] * 255
    h_pred = h_pred.astype('uint8')
    for i in range(3):
        if i == 0:
            frame_time = frame_time1
            image = image1
        elif i == 1:
            frame_time = frame_time2
            image = image2
        elif i == 2:
            frame_time = frame_time3
            image = image3

        if np.amax(h_pred[i, ...]) <= 0:
            f.write(str(count) + ',0,0,0,' + frame_time + '\n')
            out.write(image)
        else:
            (cnts, _) = cv2.findContours(h_pred[i, ...], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = np.argmax([r[2] * r[3] for r in rects])
            target = rects[max_area_idx]
            cx_pred, cy_pred = int(ratio * (target[0] + target[2] / 2)), int(ratio * (target[1] + target[3] / 2))

            f.write(str(count) + ',1,' + str(cx_pred) + ',' + str(cy_pred) + ',' + frame_time + '\n')
            image_cp = np.copy(image)
            cv2.circle(image_cp, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
            out.write(image_cp)

        count += 1

    success, image1 = cap.read()
    frame_time1 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
    success, image2 = cap.read()
    frame_time2 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
    success, image3 = cap.read()
    frame_time3 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))

f.close()
out.release()
end = time.time()
print('Prediction time:', end - start, 'secs')
print('Done......')
