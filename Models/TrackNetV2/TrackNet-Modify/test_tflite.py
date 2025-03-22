import argparse
import numpy as np
import cv2
import os
import sys
import tflite_runtime.interpreter as tflite

HEIGHT = 288
WIDTH = 512
FRAME_STACK = 3

def process_video(video_path, tflite_model_path):
    interpreter = tflite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not os.path.isfile(video_path) or not video_path.endswith('.mp4'):
        print("Not a valid video path! Please modify the --video_path argument.")
        sys.exit(1)
    else:
        # acquire video info
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.split(video_path)[-1][:-4]

    ratio = None
    size = None
    gray_imgs = []

    success, image = cap.read()
    if success:
        ratio = image.shape[0] / HEIGHT
        size = (int(WIDTH * ratio), int(HEIGHT * ratio))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name + '_predict.mp4', fourcc, fps, size)

    frame_no = 0
    while success:
        if frame_no == n_frames - 2:
            break

        success, image1 = cap.read()
        success, image2 = cap.read()
        success, image3 = cap.read()
        if not (success and image1 is not None and image2 is not None and image3 is not None):
            break

        img_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        img_gray1 = cv2.resize(img_gray1, (WIDTH, HEIGHT))

        img_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        img_gray2 = cv2.resize(img_gray2, (WIDTH, HEIGHT))

        img_gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
        img_gray3 = cv2.resize(img_gray3, (WIDTH, HEIGHT))

        input = np.stack((img_gray1, img_gray2, img_gray3), axis=-1)
        input = input[np.newaxis, ...]
        input = input.astype('float') / 255.0

        if input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            input = input / input_scale + input_zero_point

        input = input.astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], input)

        # Run inference
        interpreter.invoke()

        y_pred = interpreter.get_tensor(output_details[0]['index'])
        if output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = input_details[0]["quantization"]
            y_pred = y_pred * output_scale + output_zero_point

        y_pred = y_pred[0] > 0.5
        y_pred = y_pred.astype(np.uint8) * 255

        for i in range(3):
            if i == 0:
                image = image1
            elif i == 1:
                image = image2
            else:
                image = image3
            heatmap = y_pred[..., i]
            heatmap = heatmap.astype(np.uint8)
            if np.amax(heatmap) <= 0:
                out.write(image)
            else:
                cnts, hierarchy = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                if rects:
                    max_area_idx = 0
                    max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                    for j in range(1, len(rects)):
                        area = rects[j][2] * rects[j][3]
                        if area > max_area:
                            max_area_idx = j
                            max_area = area
                    target = rects[max_area_idx]
                    (cx_pred, cy_pred) = (int(ratio * (target[0] + target[2] / 2)), int(ratio * (target[1] + target[3] / 2)))
                    cv2.circle(image, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                out.write(image)
        frame_no += 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TFLite model on video.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--tflite_model_path', type=str, required=True, help="Path to the TFLite model file.")
    
    args = parser.parse_args()

    process_video(args.video_path, args.tflite_model_path)
