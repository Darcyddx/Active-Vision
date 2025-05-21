package com.example.activevision.trackers;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Pair;

import com.example.activevision.data.Bbox;
import com.example.activevision.data.KeyPoint;
import com.example.activevision.tflite_helpers.AIHubDefaults;
import com.example.activevision.tflite_helpers.TFLiteHelpers;
import com.example.activevision.utils.ImageOps;

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
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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

    private final int outputHeight;

    private final int outputWidth;

    private final int numJoints;

    private final ImageProcessor imageProcessor;

    private final float INPUT_SCALE;

    private final int INPUT_ZERO_POINT;

    private final float OUTPUT_SCALE;

    private final int OUTPUT_ZERO_POINT;

    private final float confThreshold = 0.25f;

    private final float keypointConfThreshold = 0.3f;



//    private final byte[] inputByteArray;
//
//    private final float[] inputFloatArray;
//
//
//    private final ByteBuffer inputByteBuffer;

    private final float scoreThresh = 0.3f;

    private int cameraCapturedWidth = 0;

    private int cameraCapturedHeight = 0;

    /**
     * Buffer to store the most recent 30 frames of keypoint sequences.
     */
    private final List<float[]> sequenceBuffer = new ArrayList<>();

    private float[] lastProbabilities = null;

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

        OUTPUT_SCALE = outputTensor.quantizationParams().getScale();
        OUTPUT_ZERO_POINT = outputTensor.quantizationParams().getZeroPoint();
        assert outputShape.length == 4;
        assert outputType == DataType.UINT8 || outputType == DataType.INT8 | outputType == DataType.FLOAT32; // U/INT8 (Quantized) and FP32 Output Supported

        inputHeight = inputShape[1];
        inputWidth = inputShape[2];
        inputChannels = inputShape[3];

        outputHeight = outputShape[1];
        outputWidth = outputShape[2];
        numJoints = outputShape[3];

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

//        if (inputType == DataType.UINT8) {
//            inputByteBuffer = ByteBuffer.allocateDirect(inputHeight * inputWidth * inputChannels);
//            inputByteBuffer.order(ByteOrder.nativeOrder());
//            inputByteArray = new byte[inputHeight * inputWidth * inputChannels];
//            inputFloatArray = null;
//        } else {
//            inputByteBuffer = ByteBuffer.allocateDirect(inputHeight * inputWidth * inputChannels * 4);
//            inputByteBuffer.order(ByteOrder.nativeOrder());
//            inputFloatArray = new float[inputHeight * inputWidth * 3];
//            inputByteArray = null;
//        }

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

    public List<float[]> postprocess(float[] output, int inputImgHeight, int inputImgWidth) {

        // Calculate the scaling factor to map model output back to original image dimensions
        float r = Math.min(inputShape[1] * 1.0f / inputImgHeight, inputShape[2] * 1.0f / inputImgWidth);
        // Calculate padding applied during letterboxing to adjust predictions back to the original image coordinates
        int widthPadding = (int) Math.round((inputShape[2] - inputImgWidth * r) / 2.0f - 0.1);
        int heightPadding = (int) Math.round((inputShape[1] - inputImgHeight * r) / 2.0f - 0.1);
        float heightPadRatio = (float) heightPadding / inputShape[1];
        float widthPadRatio = (float) widthPadding / inputShape[2];

        // output: flat array of size 1 x 47 x 8400 = 394800
        int numDetections = 8400;
        int numFeatures = 47;

        // Reshape to [8400][47]
        float[][] detections = new float[numDetections][numFeatures];
        for (int i = 0; i < numDetections; i++) {
            for (int j = 0; j < numFeatures; j++) {
                detections[i][j] = output[i * numFeatures + j];
            }
        }

        // Filter detections by confidence threshold
        List<float[]> filteredDetections = new ArrayList<>();
        for (float[] det : detections) {
            if (det[4] > confThreshold) {
                filteredDetections.add(det);
            }
        }

        // Extract best keypoint per type based on confidence
        float[][] bestKeypoints = new float[14][2];
        float[] bestConfidence = new float[14];
        Arrays.fill(bestConfidence, -1.0f);  // Initialize to -1

        for (float[] det : filteredDetections) {
            for (int i = 0; i < 14; i++) {
                float x = det[5 + i * 3];
                float y = det[5 + i * 3 + 1];
                float kconf = det[5 + i * 3 + 2];

                if (kconf > keypointConfThreshold && kconf > bestConfidence[i]) {
                    bestConfidence[i] = kconf;

                    float xScaled = x * inputShape[2]; // x * 640
                    float yScaled = y * inputShape[1]; // y * 640

                    // Remove padding
                    xScaled -= widthPadding;
                    yScaled -= heightPadding;

                    // Scale back to original image dimensions
                    xScaled /= r;
                    yScaled /= r;

                    // Clamp to image bounds
                    xScaled = Math.max(0, Math.min(xScaled, inputImgWidth));
                    yScaled = Math.max(0, Math.min(yScaled, inputImgHeight));

                    bestKeypoints[i][0] = xScaled;
                    bestKeypoints[i][1] = yScaled;
                }
            }
        }

        // Convert to List<float[]> for compatibility with resolveSymmetricKeypoints
        List<float[]> keypointsList = new ArrayList<>();
        boolean hasValidKeypoints = false;
        for (float[] kp : bestKeypoints) {
            keypointsList.add(kp);
            if (kp[0] != 0 || kp[1] != 0) {
                hasValidKeypoints = true;
            }
        }

        // Return null if no valid keypoints
        if (!hasValidKeypoints) {
            return null;
        }

        // Apply symmetry resolution if needed
        return resolveSymmetricKeypoints(keypointsList);
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