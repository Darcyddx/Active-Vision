import numpy as np
import logging
import sys, getopt
import os
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from TrackNet import ResNet_Track
# from tensorflow.keras import optimizers
from tensorflow.keras.activations import *
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW, Adadelta
from focal_loss import BinaryFocalLoss
import cv2
import math
import matplotlib.pyplot as plt
import gc


# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Setup logging
logging.basicConfig(
    filename='tracknet-grayscale.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

BATCH_SIZE=3
HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5


#Return the numbers of true positive, true negative, false positive and false negative
def outcome(y_pred, y_true, tol):
    n = y_pred.shape[0]
    i = 0
    TP = TN = FP1 = FP2 = FN = 0
    while i < n:
        for j in range(3):
            if np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) == 0:
                TN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) == 0:
                FP2 += 1
            elif np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) > 0:
                FN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) > 0:
                h_pred = y_pred[i][j] * 255
                h_true = y_true[i][j] * 255
                h_pred = h_pred.astype('uint8')
                h_true = h_true.astype('uint8')
                #h_pred
                (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

                #h_true
                (cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
                dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                if dist > tol:
                    FP1 += 1
                else:
                    TP += 1
        i += 1
    return (TP, TN, FP1, FP2, FN)

def calc_metric(TP, TN, FP1, FP2, FN):
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
        accuracy = 0
    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = 0
    return accuracy, precision, recall, f1


try:
    (opts, args) = getopt.getopt(sys.argv[1:], '', [
        'load_weights=',
        'save_weights=',
        'dataDir=',
        'epochs=',
        'tol='
    ])
    if len(opts) < 4:
        raise ''
except:
    print('usage: python3 train.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
    print('argument --load_weights is required only if you want to retrain the model')
    exit(1)

paramCount={
    'load_weights': 0,
    'save_weights': 0,
    'dataDir': 0,
    'epochs': 0,
    'tol': 0
}

for (opt, arg) in opts:
    if opt == '--load_weights':
        paramCount['load_weights'] += 1
        load_weights = arg
    elif opt == '--save_weights':
        paramCount['save_weights'] += 1
        save_weights = arg
    elif opt == '--dataDir':
        paramCount['dataDir'] += 1
        dataDir = arg
    elif opt == '--epochs':
        paramCount['epochs'] += 1
        epochs = int(arg)
    elif opt == '--tol':
        paramCount['tol'] += 1
        tol = int(arg)
    else:
        print('usage: python3 train.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
        print('argument --load_weights is required only if you want to retrain the model')
        exit(1)

if paramCount['save_weights'] == 0 or paramCount['dataDir'] == 0 or paramCount['epochs'] == 0 or paramCount['tol'] == 0:
    print('usage: python3 train.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
    print('argument --load_weights is required only if you want to retrain the model')
    exit(1)

#Loss function
def custom_loss(y_true, y_pred): #hm_true, hm_pred
   
    #loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))

    return (loss)

#Training for the first time
if paramCount['load_weights'] == 0:
    model=ResNet_Track(input_shape=(HEIGHT, WIDTH, 3))
    model.build(input_shape=(None, HEIGHT, WIDTH, 3))
    model.summary()
    # ADADELTA = Adadelta(lr=1.0)
    adamw = AdamW(learning_rate=0.001, weight_decay=0.0005)
    # model.compile(loss=custom_loss, optimizer=adamw, metrics=['accuracy'])
    model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=adamw, metrics=[tf.keras.metrics.BinaryAccuracy()]) 
# load trained model for fine-tune
else:
    # model_channel_first = load_model(load_weights, custom_objects={'custom_loss':custom_loss})
    model=ResNet_Track(input_shape=(HEIGHT, WIDTH, 3))
    model.build(input_shape=(None, HEIGHT, WIDTH, 3))
    model.load_weights(load_weights)
    # model = tfmot.quantization.keras.quantize_model(base_model)
    ADADELTA = Adadelta(lr=0.5)
    # model.compile(loss=custom_loss, optimizer=ADADELTA, metrics=['accuracy'])
    model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=optimizer, metrics=[tensorflow.keras.metrics.BinaryAccuracy()]) 
    model.summary()
    print("loaded pre-trained model")

r = os.path.abspath(os.path.join(dataDir))
path = glob(os.path.join(r, '*.npz'))
num = len(path) // 2
idx = np.arange(num, dtype='int') + 1

best_f1_score = 0
best_weights_path = None
train_idx, eval_idx = train_test_split(np.arange(num, dtype='int') + 1, test_size=0.2, random_state=42)

print('Beginning training......')
loss_list = []
for epoch in range(epochs):
    print(f"============ Epoch {epoch + 1} ================")
    loss = 0
    np.random.shuffle(train_idx)

    # Training loop
    for j in train_idx:
        data = np.load(os.path.abspath(os.path.join(dataDir, f"data_{j}.npz")))
        x_train = data["x_data"]
        y_train = data["y_data"]
  
        # Train the model
        history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
        loss += history.history["loss"][0]
        del x_train, y_train
        gc.collect()

    loss_list.append(loss)

    # Evaluation loop
    TP = TN = FP1 = FP2 = FN = 0
    for j in eval_idx:
        # Load evaluation data (channel-first format)
        data = np.load(os.path.abspath(os.path.join(dataDir, f"data_{j}.npz")))
        x_eval = data["x_data"]
        y_eval = data["y_data"]
        y_eval = np.transpose(y_eval, (0, 3, 1, 2))  # Convert to channel-first (NCHW)
        
        # Predict and calculate metrics
        y_pred = model.predict(x_eval, batch_size=BATCH_SIZE)
        y_pred = (y_pred > 0.5).astype("float32")
        y_pred = np.transpose(y_pred, (0, 3, 1, 2))  # Convert to channel-first (NCHW)
        # print("y_pred shape for eval", y_pred.shape)
        # print("y_eval shape for eval", y_eval.shape)
        tp, tn, fp1, fp2, fn = outcome(y_pred, y_eval, tol)
        TP += tp
        TN += tn
        FP1 += fp1
        FP2 += fp2
        FN += fn

        del x_eval, y_eval, y_pred
        gc.collect()

    # Print metrics for the current epoch
    print(f"Outcome of training data for epoch {epoch + 1}:")
    print("loss:", loss)
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP1): {FP1}")
    print(f"False Positives (FP2): {FP2}")
    print(f"False Negatives (FN): {FN}")

    acc, prec, recall, f1 = calc_metric(TP, TN, FP1, FP2, FN)
    print(f"F1 Score for epoch {epoch + 1}: {f1:.4f}")

    logger.info(f"Outcome of training data of epoch {epoch + 1}:")
    logger.info(f"loss: {loss}")
    logger.info(f"Number of true positives: {TP}")
    logger.info(f"Number of true negatives: {TN}")
    logger.info(f"Number of false positives FP1: {FP1}")
    logger.info(f"Number of false positives FP2: {FP2}")
    logger.info(f"Number of false negatives: {FN}")
    logger.info(f"F1 Score for epoch {epoch + 1}: {f1:.4f}")

    # Save weights if F2 score is the best
    if f1 > best_f1_score:
        best_f1_score = f1
        best_weights_path = save_weights + 'resnet_tracknetv2_best'
        model.save(best_weights_path, include_optimizer=False)
        print(f"New best F1 score: {f1:.4f}. Model saved to {best_weights_path}")
        logger.info(f"New best F1 score: {f1:.4f}. Model saved to {best_weights_path}")

print('Saving weights......')
logger.info('Saving final weights...')
last_weights_path = save_weights + 'resnet_tracknetv2_last.weight'
model.save_weights(last_weights_path)
print(f"Final weights saved to {last_weights_path}")
logger.info(f"Final weights saved to {last_weights_path}")
print('Done......')

#############################
title = 'tracknetv2-resnet'
plt.title(title)
plt.xlabel('epoch')
plt.ylabel('loss')
x = np.arange(1,epochs+1,1)
plt.plot(x, loss_list)
plt.savefig(title + '.jpg')
#############################

print('Done......')
