package com.example.active_vision_qualcomm.Trackers;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.text.InputType;
import android.util.Pair;

import com.example.active_vision_qualcomm.data.Bbox;
import com.example.active_vision_qualcomm.tflite_helpers.AIHubDefaults;
import com.example.active_vision_qualcomm.tflite_helpers.TFLiteHelpers;
import com.example.active_vision_qualcomm.utils.ImageOps;

import org.opencv.osgi.OpenCVNativeLoader;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.DequantizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * PlayerDetector is responsible for detecting players in camera frames with YOLOv8-Nano.
 * It handles model loading, input preprocessing, performing inference, and postprocessing to extract bounding boxes.
 * The class supports different data types for inputs and outputs, including quantized UINT8 and FLOAT32.
 * This implementation is adopt from
 * https://github.com/surendramaran/YOLOv8-TfLite-Object-Detector/blob/main/app/src/main/java/com/surendramaran/yolov8tflite/Detector.kt
 */
public class PlayerDetector implements AutoCloseable {
    private static final String TAG = "PlayerDetector";

    private final float STD = 255.0f;
    private final float MEAN = 0.0f;

    private final float CONF_THRES = 0.5f;
    private final float IOU_THRES = 0.5f;
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;

    private final int[] inputShape;

    private final int[] outputShape;
    private final DataType inputType;
    private final DataType outputType;

    private final ImageProcessor imageProcessor;

    private final float INPUT_SCALE;

    private final int INPUT_ZERO_POINT;

    private final float OUTPUT_SCALE;

    private final int OUTPUT_ZERO_POINT;

    private final TensorBuffer outputBuffer;

    public PlayerDetector(Context context,
                         String modelPath) throws IOException, NoSuchAlgorithmException {
        this(context, modelPath,  AIHubDefaults.delegatePriorityOrder);
    }

    /**
     * Create a player detector with Yolov8-nano for detecting players in camera frames
     * @param context App context.
     * @param modelPath Model path to load.
     * @param delegatePriorityOrder The priority order of delegates to be applied for model inference.
     * @throws IOException If the model can't be read from disk.
     * @throws NoSuchAlgorithmException
     */
    public PlayerDetector(Context context,
                         String modelPath,
                         TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {

        // Load TF Lite model
        Pair<MappedByteBuffer, String> modelAndHash = TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);
        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                modelAndHash.first,
                delegatePriorityOrder,
                AIHubDefaults.numCPUThreads,
                context.getApplicationInfo().nativeLibraryDir,
                context.getCacheDir().getAbsolutePath(),
                modelAndHash.second
        );
        tfLiteInterpreter = iResult.first;
        tfLiteDelegateStore = iResult.second;
        // Validate TF Lite model fits requirements for this app
        assert tfLiteInterpreter.getInputTensorCount() == 1;
        Tensor inputTensor = tfLiteInterpreter.getInputTensor(0);
        inputShape = inputTensor.shape();
        inputType = inputTensor.dataType();
        INPUT_SCALE = inputTensor.quantizationParams().getScale();
        INPUT_ZERO_POINT= inputTensor.quantizationParams().getZeroPoint();

        assert inputShape.length == 4; // 4D Input Tensor: [Batch, Height, Width, Channels]
        assert inputShape[0] == 1; // Batch size is 1
        assert inputShape[3] == 3; // Input tensor should have 3 channels
        assert inputType == DataType.UINT8 || inputType == DataType.INT8 || inputType == DataType.FLOAT32; // INT8 (Quantized) and FP32 Input Supported

        assert tfLiteInterpreter.getOutputTensorCount() == 1;
        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        outputShape = outputTensor.shape();
        outputType = outputTensor.dataType();
        OUTPUT_SCALE = outputTensor.quantizationParams().getScale();
        OUTPUT_ZERO_POINT = outputTensor.quantizationParams().getZeroPoint();
        assert outputShape.length == 3; // 3D Output Tensor: [Batch, num_features(x, y, w, h, obj_0_conf, obj_1_conf, ...), num_predictions]
        assert outputType == DataType.UINT8 || outputType == DataType.INT8 | outputType == DataType.FLOAT32; // U/INT8 (Quantized) and FP32 Output Supported
        // Set-up preprocessor
        if (inputType == DataType.FLOAT32) {
            imageProcessor = new ImageProcessor.Builder()
                    .add(new NormalizeOp(MEAN, STD))
                    .build();
        } else {
            imageProcessor = new ImageProcessor.Builder()
                    .add(new NormalizeOp(MEAN, STD))
                    .add(new QuantizeOp(INPUT_ZERO_POINT, INPUT_SCALE))
                    .add(new CastOp(inputType))
                    .build();
        }
        // TensorBuffer to hold the model's output data
        outputBuffer = TensorBuffer.createFixedSize(outputShape, outputType);
    }

    @Override
    public void close() throws Exception {
        tfLiteInterpreter.close();
        for (Delegate delegate: tfLiteDelegateStore.values()) {
            delegate.close();
        }
    }

    /**
     * Preprocesses a Bitmap image to prepare it for model inference
     * This includes resizing, letterboxing, normalization
     * @param image The Bitmap image to preprocess.
     * @return A ByteBuffer containing the preprocessed input data ready for inference.
     */
    public ByteBuffer preprocess(Bitmap image) {
        Bitmap letterboxedImage;
        // Apply letterboxing to maintain aspect ratio and fit the model's input size
        if (image.getHeight() != inputShape[1] || image.getWidth() != inputShape[2]) {
            letterboxedImage = ImageOps.letterbox(image, new Pair<>(inputShape[1], inputShape[2]), false, false, true, 32);
        } else {
            letterboxedImage = image;
        }

        // Convert type and fill input buffer
        ByteBuffer inputBuffer;
        TensorImage tImg = TensorImage.fromBitmap(letterboxedImage);
        inputBuffer = imageProcessor.process(tImg).getBuffer();
        return inputBuffer;
    }

    /**
     * perform model inferencing with TFLite interpreter
     * @param inputBuffer The ByteBuffer containing the preprocessed input data.
     * @return  A TensorBuffer containing the output data from the model.
     */
    public TensorBuffer inference(ByteBuffer inputBuffer) {
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(
                outputShape,
                outputType
        );

        if (tfLiteInterpreter != null) {
            tfLiteInterpreter.run(inputBuffer, outputBuffer.getBuffer());
        }
        if (outputType == DataType.UINT8) {
            TensorProcessor tensorProcessor = new TensorProcessor.Builder()
                    .add(new DequantizeOp(OUTPUT_ZERO_POINT, OUTPUT_SCALE))
                    .build();
            outputBuffer = tensorProcessor.process(outputBuffer);
        }
        return outputBuffer;
    }

    /**
     * Postprocesses the model's output to extract bounding boxes of detected players.
     * This involves interpreting the model's predictions, applying confidence thresholds,
     * and performing NMS to eliminate redundant bounding boxes
     *
     * @param outputArray    The flattened output data from the model as a float array.
     *                       output format for single class:[x1, x2, ..., x_n, y1, y2, ..., y_n, w1, w2, ..., w_n, h1, h2, ..., h_n,
     *                       conf_cls0_0, conf_cls0_1, ..., conf_cls0_n]
     * @param inputImgHeight The original input image's height in pixels.
     * @param inputImgWidth  The original input image's width in pixels.
     * @return A list of Bbox objects representing detected players. Returns null if no detections are found.
     */
    public List<Bbox> postprocess(float[] outputArray, int inputImgHeight, int inputImgWidth) {
        ArrayList<Bbox> bboxes = new ArrayList<>();
        int numFeatures = outputShape[1];
        int numPreds = outputShape[2];
        // Calculate the scaling factor to map model output back to original image dimensions
        float r = Math.min(inputShape[1] * 1.0f / inputImgHeight, inputShape[2] * 1.0f / inputImgWidth);
        // Calculate padding applied during letterboxing to adjust predictions back to the original image coordinates
        int widthPadding = (int) Math.round((inputShape[2] - inputImgWidth * r) / 2.0f - 0.1);
        int heightPadding = (int) Math.round((inputShape[1] - inputImgHeight * r) / 2.0f - 0.1);
        float heightPadRatio = (float) heightPadding / inputShape[1];
        float widthPadRatio = (float) widthPadding / inputShape[2];
        for (int i = 0; i < numPreds; i++) {
            float maxConf = CONF_THRES;
            int maxIdx = -1;
            int j = 4;
            int arrIdx = i + numPreds * j;
            // Iterate through the confidence scores for each prediction
            // ignore the confidence score if < maxConf and continue updating the maximum confidence if a higher value is found
            // track the index of the highest confidence feature
            while (j < numFeatures) {
                if (outputArray[arrIdx] > maxConf) {
                    maxConf = outputArray[arrIdx];
                    maxIdx = j - 4;
                }
                j++;
                arrIdx += numPreds;
            }

            // If the highest confidence exceeds the threshold, consider it a valid detection
            if (maxConf > CONF_THRES) {
                float cx = outputArray[i] - widthPadRatio;
                float cy = outputArray[i + numPreds] - heightPadRatio;
                float w = outputArray[i + numPreds * 2];
                float h = outputArray[i + numPreds * 3];
                // Calculate the top-left and bottom-right coordinates of the bounding box
                float x1 = cx - (w / 2.0f);
                float y1 = cy - (h / 2.0f);
                float x2 = cx + (w / 2.0f);
                float y2 = cy + (h / 2.0f);
                if (x1 < 0 || x1 > 1) continue;
                if (y1 < 0 || y1 > 1) continue;
                if (x2 < 0 || x2 > 1) continue;
                if (y2 < 0 || y2 > 1) continue;
                bboxes.add(new Bbox(maxIdx, maxConf, cx, cy, w, h, new RectF(x1, y1, x2, y2)));
            }
        }
        if (bboxes.isEmpty()) {
            return null;
        }
        // Apply NMS to remove redundant overlapping bounding boxes
        return applyNMS(bboxes);
    }

    /**
     * Apply NMS, which selects the bounding boxes with the highest confidence scores and removes others that have
     * a high IoU with the selected boxes.
     * @param boxes The list of bounding boxes to process.
     * @return A list of bounding boxes after NMS has been applied.
     */
    private List<Bbox> applyNMS(List<Bbox> boxes) {
        // Step 1: Sort the bounding boxes in descending order based on their confidence scores.
        boxes.sort(new Comparator<Bbox>() {
            @Override
            public int compare(Bbox b1, Bbox b2) {
                return Float.compare(b2.getCnf(), b1.getCnf());
            }
        });
        List<Bbox> selectedBoxes = new ArrayList<>();
        // Step 2: Iterate through the sorted bounding boxes.
        // At each iteration, select the box with the highest confidence score,
        // add it to the list of selected boxes, and remove it from the original list.
        while (!boxes.isEmpty()) {
            Bbox first = boxes.get(0);
            selectedBoxes.add(first);
            boxes.remove(0);

            Iterator<Bbox> iterator = boxes.iterator();

            // Step 3: Compare the selected box with the remaining boxes to calculate IoU.
            // If the IoU exceeds the predefined threshold, remove the overlapping box.
            while (iterator.hasNext()) {
                Bbox nextBox = iterator.next();
                float iou = calculateIoU(first, nextBox);
                if (iou >= IOU_THRES) {
                    iterator.remove();
                }
            }
        }
        return selectedBoxes;
    }

    private float calculateIoU(Bbox box1, Bbox box2) {
        float x1 = Math.max(box1.getRect().left, box2.getRect().left);
        float y1 = Math.max(box1.getRect().top, box2.getRect().top);
        float x2 = Math.min(box1.getRect().right, box2.getRect().right);
        float y2 = Math.min(box1.getRect().bottom, box2.getRect().bottom);
        float intersectionArea = Math.max(0F, x2 - x1) * Math.max(0F, y2 - y1);
        float box1Area = box1.getWidth() * box1.getHeight();
        float box2Area = box2.getWidth() * box2.getHeight();
        return intersectionArea / (box1Area + box2Area - intersectionArea);
    }


}
