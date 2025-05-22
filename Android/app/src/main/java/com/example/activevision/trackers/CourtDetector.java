package com.example.activevision.trackers;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.util.Pair;

import com.example.activevision.tflite_helpers.AIHubDefaults;
import com.example.activevision.tflite_helpers.TFLiteHelpers;
import com.example.activevision.utils.ImageOps;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.ops.CastOp;
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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class CourtDetector  implements AutoCloseable {

    private static final String TAG = "CourtDetector";

    private final Interpreter tfLiteInterpreter;

    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;

    private final int[] inputShape;

    private final int[] outputShape;
    private final DataType inputType;
    private final DataType outputType;

    private final int inputHeight;
    private final int inputWidth;
    private final int inputChannels;


    private final ImageProcessor imageProcessor;

    private final float INPUT_SCALE;

    private final int INPUT_ZERO_POINT;

    private final float confThreshold = 0.25f;

    private final float keypointConfThreshold = 0.3f;

    /**
     * The result of the last calculation is the probability results.
     * @param context Application context.
     * @param modelPath Path to the TFLite model file.
     * @param delegatePriorityOrder Priority order of TFLite delegates to use.
     * @throws IOException If there is an error reading the model file.
     */
    public CourtDetector(Context context,
                         String modelPath,
                         TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException  {

        // Load the TFLite model file
        // Assume the model file is stored in assets, and use the TFLiteHelpers utility to load it
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

        // INPUT
        inputHeight = inputShape[1];     // should be 640
        inputWidth = inputShape[2];      // should be 640
        inputChannels = inputShape[3];   // should be 3

        assert outputShape.length == 3;
        assert outputType == DataType.FLOAT32; // Only FP32 supported by your model

        if (inputType == DataType.FLOAT32) {
            imageProcessor = new ImageProcessor.Builder()
                    .add(new NormalizeOp(0.0f, 255.0f))
                    .build();
        } else {
            imageProcessor = new ImageProcessor.Builder()
                    .add(new NormalizeOp(0.0f, 255.0f))
                    .add(new QuantizeOp(INPUT_ZERO_POINT, INPUT_SCALE))
                    .add(new CastOp(inputType))
                    .build();
        }

        Log.d(TAG, "Court detector load successful");

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

        Log.d(TAG, "Output buffer shape: " + Arrays.toString(outputBuffer.getShape()));

        return outputBuffer;
    }

    public List<float[]> postprocess(float[] output, int inputImgHeight, int inputImgWidth) {
        int numDetections = 8400;
        int numFeatures = 47;  // 4 bbox + 1 objectness + 14*3 keypoints

        // Reshape output [1, 47, 8400] -> [8400][47]
        float[][] detections = new float[numDetections][numFeatures];
        for (int i = 0; i < numDetections; i++) {
            for (int j = 0; j < numFeatures; j++) {
                detections[i][j] = output[j * numDetections + i];
            }
        }

        // Filter detections by objectness confidence
        float confThreshold = 0.3f;
        float keypointConfThreshold = 0.5f;
        List<float[]> filteredDetections = new ArrayList<>();
        for (float[] det : detections) {
            if (det[4] > confThreshold) {
                filteredDetections.add(det);
            }
        }

        // Extract all keypoints (x,y) scaled to 640 x 640 for each detection
        List<float[][]> allKeypoints = new ArrayList<>();
        List<float[]> confidencesList = new ArrayList<>();

        for (float[] det : filteredDetections) {
            float[][] keypoints = new float[14][2];
            float[] confidences = new float[14];
            for (int i = 0; i < 14; i++) {
                float x = det[5 + i * 3];
                float y = det[5 + i * 3 + 1];
                float kconf = det[5 + i * 3 + 2];
                confidences[i] = kconf;

                if (kconf > keypointConfThreshold) {
                    keypoints[i][0] = x * 640f;
                    keypoints[i][1] = y * 640f;
                } else {
                    keypoints[i][0] = 0f;
                    keypoints[i][1] = 0f;
                }
            }
            allKeypoints.add(keypoints);
            confidencesList.add(confidences);
        }

        if (allKeypoints.isEmpty()) {
            return null;
        }

        // Convert List<float[]> confidencesList to array for argmax
        float[][] confidencesArray = new float[confidencesList.size()][14];
        for (int i = 0; i < confidencesList.size(); i++) {
            confidencesArray[i] = confidencesList.get(i);
        }

        // Find best detection index per keypoint (argmax over detections)
        int[] bestIndices = new int[14];
        for (int kp = 0; kp < 14; kp++) {
            float maxConf = -1f;
            int maxIdx = -1;
            for (int detIdx = 0; detIdx < confidencesArray.length; detIdx++) {
                if (confidencesArray[detIdx][kp] > maxConf) {
                    maxConf = confidencesArray[detIdx][kp];
                    maxIdx = detIdx;
                }
            }
            bestIndices[kp] = maxIdx;
        }

        // Pick the best keypoints for each type by confidence
        List<float[]> bestKeypoints = new ArrayList<>();
        for (int kp = 0; kp < 14; kp++) {
            int detIdx = bestIndices[kp];
            float[] kpCoord = allKeypoints.get(detIdx)[kp];
            bestKeypoints.add(kpCoord);
        }

        // apply symmetry resolution
        List<float[]> adjustedKeypoints = resolveSymmetricKeypoints(bestKeypoints);

//        List<float[]> scaledKeypoints = new ArrayList<>();
//
//        // scale back to image original height and width
//        float scaleX = (float) inputImgWidth / inputWidth;
//        float scaleY = (float) inputImgHeight / inputHeight;
//
//        for (float[] kp : adjustedKeypoints) {
//            float x = kp[0] * scaleX;
//            float y = kp[1] * scaleY;
//
//            scaledKeypoints.add(new float[]{x, y});
//        }

        return scaleKeypointsWithPadding(adjustedKeypoints, inputImgWidth, inputImgHeight);
    }

    public List<float[]> scaleKeypointsWithPadding(List<float[]> adjustedKeypoints,
                                                   int inputImgWidth, int inputImgHeight) {
        List<float[]> scaledKeypoints = new ArrayList<>();

        // Step 1: Calculate the scale ratio used during resizing
        float r = Math.min((float) inputWidth / inputImgWidth,
                (float) inputHeight / inputImgHeight);

        // Step 2: Compute the size of the image after resizing but before padding
        int newUnpadWidth = Math.round(inputImgWidth * r);
        int newUnpadHeight = Math.round(inputImgHeight * r);

        // Step 3: Compute the padding added to width and height
        float dw = (inputWidth - newUnpadWidth) / 2.0f;  // width padding
        float dh = (inputHeight - newUnpadHeight) / 2.0f; // height padding

        // Step 4: Undo the padding and scale back to original image size
        for (float[] kp : adjustedKeypoints) {
            float x = (kp[0] - dw) / r;
            float y = (kp[1] - dh) / r;

            // Ensure the keypoint is within image bounds
            x = Math.max(0, Math.min(x, inputImgWidth - 1));
            y = Math.max(0, Math.min(y, inputImgHeight - 1));

            scaledKeypoints.add(new float[]{x, y});
        }

        return scaledKeypoints;
    }


    public List<float[]> resolveSymmetricKeypoints(List<float[]> kp) {
        List<float[]> resolved = new ArrayList<>();
        for (float[] point : kp) {
            resolved.add(point.clone());
        }

        float center = (kp.get(5)[0] + kp.get(8)[0]) / 2.0f;

        List<Integer> leftIds = Arrays.asList(0, 1, 4, 7, 10, 11);
        List<Integer> rightIds = Arrays.asList(2, 3, 6, 9, 12, 13);
        Set<Integer> incorrectIds = new HashSet<>();

        for (int idx : leftIds) {
            if (resolved.get(idx)[0] >= center) {
                incorrectIds.add(idx);
            }
        }
        for (int idx : rightIds) {
            if (resolved.get(idx)[0] <= center) {
                incorrectIds.add(idx);
            }
        }

        // Swap (4, 6)
        if (incorrectIds.contains(4) && incorrectIds.contains(6)) {
            Collections.swap(resolved, 4, 6);
            incorrectIds.remove(4);
            incorrectIds.remove(6);
        }

        // Swap (7, 9)
        if (incorrectIds.contains(7) && incorrectIds.contains(9)) {
            Collections.swap(resolved, 7, 9);
            incorrectIds.remove(7);
            incorrectIds.remove(9);
        }

        // Fix (4, 6)
        if (incorrectIds.contains(4) || incorrectIds.contains(6)) {
            if (incorrectIds.contains(6)) {
                float[] v = vector(resolved.get(4), resolved.get(5));
                resolved.get(6)[0] = resolved.get(5)[0] + v[0];
                resolved.get(6)[1] = resolved.get(5)[1] + v[1];
            } else {
                float[] v = vector(resolved.get(6), resolved.get(5));
                resolved.get(4)[0] = resolved.get(5)[0] - v[0];
                resolved.get(4)[1] = resolved.get(5)[1] - v[1];
            }
        }

        // Fix (7, 9)
        if (incorrectIds.contains(7) || incorrectIds.contains(9)) {
            if (incorrectIds.contains(9)) {
                float[] v = vector(resolved.get(7), resolved.get(8));
                resolved.get(9)[0] = resolved.get(8)[0] + v[0];
                resolved.get(9)[1] = resolved.get(8)[1] + v[1];
            } else {
                float[] v = vector(resolved.get(9), resolved.get(8));
                resolved.get(7)[0] = resolved.get(8)[0] + v[0];
                resolved.get(7)[1] = resolved.get(8)[1] + v[1];
            }
        }

        // Fix (1, 2)
        if (incorrectIds.contains(1) || incorrectIds.contains(2)) {
            if (incorrectIds.contains(2) && !incorrectIds.contains(0) && !incorrectIds.contains(1)) {
                float[] p = intersect(resolved.get(6), resolved.get(9), resolved.get(0), resolved.get(1));
                resolved.get(2)[0] = p[0]; resolved.get(2)[1] = p[1];
            } else if (!incorrectIds.contains(2) && !incorrectIds.contains(3)) {
                float[] p = intersect(resolved.get(4), resolved.get(7), resolved.get(2), resolved.get(3));
                resolved.get(1)[0] = p[0]; resolved.get(1)[1] = p[1];
            }
        }

        // Fix (0, 3)
        if (incorrectIds.contains(0) || incorrectIds.contains(3)) {
            if (incorrectIds.contains(3)) {
                float[] v = vector(resolved.get(1), resolved.get(0));
                resolved.get(3)[0] = resolved.get(2)[0] + v[0];
                resolved.get(3)[1] = resolved.get(2)[1] + v[1];
            } else {
                float[] v = vector(resolved.get(3), resolved.get(2));
                resolved.get(0)[0] = resolved.get(1)[0] - v[0];
                resolved.get(0)[1] = resolved.get(1)[1] - v[1];
            }
        }

        // Fix (11, 12)
        if (incorrectIds.contains(11) || incorrectIds.contains(12)) {
            if (incorrectIds.contains(12) && !incorrectIds.contains(10) && !incorrectIds.contains(11)) {
                float[] p = intersect(resolved.get(6), resolved.get(9), resolved.get(10), resolved.get(11));
                resolved.get(12)[0] = p[0]; resolved.get(12)[1] = p[1];
            } else if (!incorrectIds.contains(12) && !incorrectIds.contains(13)) {
                float[] p = intersect(resolved.get(4), resolved.get(7), resolved.get(12), resolved.get(13));
                resolved.get(11)[0] = p[0]; resolved.get(11)[1] = p[1];
            }
        }

        // Fix (10, 13)
        if (incorrectIds.contains(10) || incorrectIds.contains(13)) {
            if (incorrectIds.contains(13)) {
                float[] v = vector(resolved.get(11), resolved.get(10));
                resolved.get(13)[0] = resolved.get(12)[0] + v[0];
                resolved.get(13)[1] = resolved.get(12)[1] + v[1];
            } else {
                float[] v = vector(resolved.get(13), resolved.get(12));
                resolved.get(10)[0] = resolved.get(11)[0] - v[0];
                resolved.get(10)[1] = resolved.get(11)[1] - v[1];
            }
        }

        return resolved;
    }

    private float[] vector(float[] p1, float[] p2) {
        return new float[] { Math.abs(p2[0] - p1[0]), Math.abs(p2[1] - p1[1]) };
    }

    private float[] intersect(float[] a1, float[] a2, float[] b1, float[] b2) {
        float[] L1 = line(a1, a2);
        float[] L2 = line(b1, b2);
        float det = L1[0] * L2[1] - L2[0] * L1[1];

        if (Math.abs(det) < 1e-5) {
            return a2;
        }

        float x = (L2[1] * L1[2] - L1[1] * L2[2]) / det;
        float y = (L1[0] * L2[2] - L2[0] * L1[2]) / det;
        return new float[] { x, y };
    }

    private float[] line(float[] p1, float[] p2) {
        float A = p2[1] - p1[1];
        float B = p1[0] - p2[0];
        float C = A * p1[0] + B * p1[1];
        return new float[] { A, B, C };
    }


    @Override
    public void close() throws Exception {
        tfLiteInterpreter.close();
        for (Delegate delegate: tfLiteDelegateStore.values()) {
            delegate.close();
        }
    }
}