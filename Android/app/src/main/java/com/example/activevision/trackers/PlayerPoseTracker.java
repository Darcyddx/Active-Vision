package com.example.activevision.trackers;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Pair;

import com.example.activevision.data.Bbox;
import com.example.activevision.data.KeyPoint;
import com.example.activevision.data.PoseInferenceInfo;
import com.example.activevision.data.PosePreprocessInfo;
import com.example.activevision.tflite_helpers.AIHubDefaults;
import com.example.activevision.tflite_helpers.TFLiteHelpers;
import com.example.activevision.utils.ImageOps;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
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
import java.util.List;
import java.util.Map;

public class PlayerPoseTracker {
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

//    private final byte[] inputByteArray;
//
//    private final float[] inputFloatArray;
//
//
//    private final ByteBuffer inputByteBuffer;

    private final float scoreThresh = 0.3f;

    private int cameraCapturedWidth = 0;

    private int cameraCapturedHeight = 0;

    public PlayerPoseTracker(Context context,
                             String modelPath,
                             TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {

        new OpenCVNativeLoader().init();
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


    public List<PosePreprocessInfo> preprocess(Bitmap bitmap, List<Bbox> bboxes) {
        cameraCapturedHeight = bitmap.getHeight();
        cameraCapturedWidth = bitmap.getWidth();
        // Convert Bitmap to OpenCV Mat (ABGR format)
        Mat inputMatAbgr = new Mat();
        Utils.bitmapToMat(bitmap, inputMatAbgr);

        Mat matBgr = new Mat();
        Imgproc.cvtColor(inputMatAbgr, matBgr, Imgproc.COLOR_BGRA2BGR);

        List<Bbox> actualBoxes = new ArrayList<>();
        int maxSide = Math.max(bitmap.getHeight(), bitmap.getWidth());
        for (Bbox bbox: bboxes) {
            actualBoxes.add(ImageOps.getActualBox(bbox, maxSide));
        }

        List<PosePreprocessInfo> posePreprocessInfos = new ArrayList<>();
        for (Bbox bbox: actualBoxes) {
            Point center = new Point(bbox.getCx(), bbox.getCy());
            Point scale = ImageOps.getBoxScale(bbox);
            Mat trans = ImageOps.getAffineTransMatrix(center, scale, 0.0f, new Point(inputWidth, inputHeight), new Point(0.0, 0.0), false);
            Mat inputMat = new Mat();
            Imgproc.warpAffine(matBgr, inputMat, trans, new Size(inputWidth, inputHeight), Imgproc.INTER_LINEAR);

            Bitmap croppedBitmap = Bitmap.createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888);

            Utils.matToBitmap(inputMat, croppedBitmap);

            Mat normalizedMat = new Mat();
            inputMat.convertTo(normalizedMat, CvType.CV_32FC3, 1.0 / 255.0);

//            Core.subtract(normalizedMat, new Scalar(0.5, 0.5, 0.5), normalizedMat);
//
//            Core.divide(normalizedMat, new Scalar(0.5, 0.5, 0.5), normalizedMat);

//            inputByteBuffer.rewind(); // Reset the buffer's position
//            if (inputType == DataType.UINT8) {
//                // perform quantization
//                Core.divide(normalizedMat, new Scalar(INPUT_SCALE, INPUT_SCALE, INPUT_SCALE), normalizedMat);
//                Core.add(normalizedMat, new Scalar(INPUT_ZERO_POINT, INPUT_ZERO_POINT, INPUT_ZERO_POINT), normalizedMat);
//                normalizedMat.convertTo(normalizedMat, CvType.CV_8UC3);
//                normalizedMat.get(0, 0, inputByteArray);
//                inputByteBuffer.put(inputByteArray);
//            } else {
//                normalizedMat.get(0, 0, inputFloatArray);
//                FloatBuffer inputFloatBuffer = inputByteBuffer.asFloatBuffer();
//                inputFloatBuffer.put(inputFloatArray);
//            }
            ByteBuffer inputByteBuffer;
            TensorImage tImg = TensorImage.fromBitmap(croppedBitmap);
            inputByteBuffer = imageProcessor.process(tImg).getBuffer();
            posePreprocessInfos.add(new PosePreprocessInfo(inputByteBuffer, center, scale));

        }
        return posePreprocessInfos;
    }

    public List<PoseInferenceInfo> inference(List<PosePreprocessInfo> posePreprocessInfos) {
        List<PoseInferenceInfo> inferenceInfos = new ArrayList<>();
        for (PosePreprocessInfo poseData: posePreprocessInfos) {
            TensorBuffer outputBuffer = TensorBuffer.createFixedSize(
                    outputShape,
                    outputType
            );

            if (tfLiteInterpreter != null) {
                tfLiteInterpreter.run(poseData.getByteBuffer(), outputBuffer.getBuffer());
            }
            if (outputType == DataType.UINT8) {
                TensorProcessor tensorProcessor = new TensorProcessor.Builder()
                        .add(new DequantizeOp(OUTPUT_ZERO_POINT, OUTPUT_SCALE))
                        .build();
                outputBuffer = tensorProcessor.process(outputBuffer);
            }
            inferenceInfos.add(new PoseInferenceInfo(outputBuffer.getFloatArray(), poseData.getCenter(), poseData.getScale()));

        }
        return inferenceInfos;
    }

    public void getMaxPreds(float[] heatmap, List<Point> preds, List<Float> maxVals) {
        for (int c = 0; c < numJoints; c++) {
            float bestVal = heatmap[c];
            int bestIdx = 0;
            int pixelCounter = 0;

            for (int y = 0; y < outputHeight; y++) {
                for (int x = 0; x < outputWidth; x++) {
                    int idx = y * (outputWidth * numJoints) + x * numJoints + c;
                    float val = heatmap[idx];
                    if (val > bestVal) {
                        bestVal = val;
                        bestIdx = pixelCounter;
                    }
                    pixelCounter++;
                }
            }

            int bestY = bestIdx / outputWidth;
            int bestX = bestIdx % outputWidth;

            preds.add(new Point(bestX, bestY));
            maxVals.add(bestVal);
        }
    }

    public void getFinalPreds(float[] heatmap, Point center, Point scale, List<Point> coords, List<Float> maxVals) {
        getMaxPreds(heatmap, coords, maxVals);
        int wh = outputHeight * outputWidth;
        for (int i = 0; i < numJoints; i++) {
            int x = (int) (Math.floor(coords.get(i).x + 0.5));
            int y = (int) (Math.floor(coords.get(i).y + 0.5));

            if ((x > 1 && x < (outputWidth - 1)) && (y > 1 && y < (outputHeight - 1))) {
                float[] channelData = new float[wh];
                for (int row = 0; row < outputHeight; row++) {
                    for (int col = 0; col < outputWidth; col++) {
                        int idx = row * (outputWidth * numJoints) + col * numJoints + i;
                        channelData[row * outputWidth + col] = heatmap[idx];
                    }
                }
                // Calculate neighbor differences
                float diffX = channelData[y * outputWidth + (x + 1)] - channelData[y * outputWidth + (x - 1)];
                float diffY = channelData[(y + 1) * outputWidth + x] - channelData[(y - 1) * outputWidth + x];

                // Apply sub-pixel shift
                coords.get(i).x += Math.signum(diffX) * 0.25f;
                coords.get(i).y += Math.signum(diffY) * 0.25f;
            }
        }
        float rot = 0f;
        Point outputSize = new Point(outputWidth, outputHeight);
        Mat trans = ImageOps.getAffineTransMatrix(center, scale, rot, outputSize, new Point(0.0, 0.0), true);
        for (int c = 0; c < numJoints; c++) {
            ImageOps.affineTransform(coords.get(c), trans);
        }
    }

    public List<List<KeyPoint>> postprocess(List<PoseInferenceInfo> modelOutputs) {
        List<List<KeyPoint>> frameKpList = new ArrayList<>();
        for (PoseInferenceInfo output: modelOutputs) {
            List<KeyPoint> kpList = new ArrayList<>();
            List<Point> coords = new ArrayList<>();
            List<Float> maxVals = new ArrayList<>();
            getFinalPreds(output.getHeatmap(), output.getCenter(), output.getScale(), coords, maxVals);
            for (int i = 0; i < coords.size(); i++) {
                KeyPoint kp = new KeyPoint();
                if (maxVals.get(i) > scoreThresh) {
                    kp.setPoint(coords.get(i));
                } else {
                    kp.setPoint(new Point(-1.0, -1.0));
                }
                kp.setScore(maxVals.get(i));
                kpList.add(kp);
            }
            frameKpList.add(kpList);
        }
        return frameKpList;
    }
}
