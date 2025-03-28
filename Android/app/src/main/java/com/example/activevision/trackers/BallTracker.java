package com.example.activevision.trackers;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Pair;

import com.example.activevision.data.BallPos;
import com.example.activevision.tflite_helpers.AIHubDefaults;
import com.example.activevision.tflite_helpers.TFLiteHelpers;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.osgi.OpenCVNativeLoader;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.DequantizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
/**
 * The BallTracker class integrates TensorFlow Lite for model inference to detect and track
 * the position of tennis balls in video frames. It leverages optimized machine learning models
 * and hardware acceleration using delegates like GPU and NPU to achieve real-time performance.
 * Author: Zhiyuan Lu
 * Date: 21/03/2025
 */
public class BallTracker implements AutoCloseable {
    private static final String TAG = "TennisTracker";

    private static final float STD = 255.0f;
    private static final float MEAN = 0.0f;
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;

    // Model input and output properties
    private final int[] inputShape;

    private final int[] outputShape;
    private final DataType inputType;
    private final DataType outputType;

    private TensorProcessor dequantProcessor = null;

    private final float INPUT_SCALE;

    private final int INPUT_ZERO_POINT;

    private final Scalar INPUT_SCALE_SCALAR;  // Scale factor for input quantization (for uint8 model)

    private final Scalar INPUT_ZERO_POINT_SCALAR; // Zero-point for input quantization

    private final float OUTPUT_SCALE;  // Scale factor for output dequantization (for uint8 model)

    private final int OUTPUT_ZERO_POINT; // Zero-point for output dequantization

    // Buffers for input data based on model's input type
    private final byte[] inputByteArray;

    private final float[] inputFloatArray;


    private final ByteBuffer inputByteBuffer;

    public BallTracker(Context context,
                         String modelPath) throws IOException, NoSuchAlgorithmException {
        this(context, modelPath,  AIHubDefaults.delegatePriorityOrder);
    }

    /**
     * Create a ball tracker analyzer (based on TrackNetv2) from the given tflite model
     * Uses default compute units: NPU, GPU, CPU
     * @param context App context.
     * @param modelPath Model path to load.
     * @param delegatePriorityOrder The priority order of delegates to be applied for model inference.
     * @throws IOException If the model can't be read from disk.
     * @throws NoSuchAlgorithmException
     */
    public BallTracker(Context context,
                         String modelPath,
                         TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {

        // Initialize OpenCV
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
        assert inputShape.length == 4; // 4D Input Tensor: [Batch, Height, Width, Channels]
        assert inputShape[0] == 1; // Batch size is 1
        assert inputShape[3] == 9; // Input tensor should have 3 channels
        // assert inputShape[3] == 9; // Input tensor should have 3 channels
        assert inputType == DataType.UINT8 || inputType == DataType.FLOAT32; // INT8 (Quantized) and FP32 Input Supported

        assert tfLiteInterpreter.getOutputTensorCount() == 1;
        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        outputShape = outputTensor.shape();
        outputType = outputTensor.dataType();
        assert outputShape.length == 4; // 4D Output Tensor: [Batch, Height, Width, Channels]
        assert inputShape[0] == 1; // Batch size is 1
        assert outputShape[3] == 3; // output 3 heatmaps
        assert outputType == DataType.UINT8 || outputType == DataType.INT8 | outputType == DataType.FLOAT32; // U/INT8 (Quantized) and FP32 Output Supported

        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];
        int inputChannels = inputShape[3];

        int outputHeight = outputShape[1];
        int outputWidth = outputShape[2];

        // Retrieve quantization parameters for input and output tensors
        INPUT_SCALE = inputTensor.quantizationParams().getScale();
        INPUT_ZERO_POINT= inputTensor.quantizationParams().getZeroPoint();
        OUTPUT_SCALE = outputTensor.quantizationParams().getScale();
        OUTPUT_ZERO_POINT = outputTensor.quantizationParams().getZeroPoint();

        INPUT_SCALE_SCALAR = new Scalar(INPUT_SCALE, INPUT_SCALE, INPUT_SCALE);
        INPUT_ZERO_POINT_SCALAR = new Scalar(INPUT_ZERO_POINT, INPUT_ZERO_POINT, INPUT_ZERO_POINT);


        if (inputType == DataType.UINT8) {
            dequantProcessor = new TensorProcessor.Builder()
                    .add(new DequantizeOp(
                            OUTPUT_ZERO_POINT,
                            OUTPUT_SCALE))
                    .build();
        }

        // Allocate re-usable memory
        if (inputType == DataType.UINT8) {
            inputByteBuffer = ByteBuffer.allocateDirect(inputHeight * inputWidth * inputChannels);
            inputByteBuffer.order(ByteOrder.nativeOrder());
            inputByteArray = new byte[inputHeight * inputWidth * inputChannels];
            inputFloatArray = null;
        } else {
            inputByteBuffer = ByteBuffer.allocateDirect(inputHeight * inputWidth * inputChannels * 4);
            inputByteBuffer.order(ByteOrder.nativeOrder());
            inputFloatArray = new float[inputHeight * inputWidth * 3];
            inputByteArray = null;
        }

    }

    @Override
    public void close() throws Exception {
        tfLiteInterpreter.close();
        for (Delegate delegate: tfLiteDelegateStore.values()) {
            delegate.close();
        }
    }

    /**
     * @return TrackNet model input width
     */
    public int getInputWidth() {
        return inputShape[2];
    }

    /**
     * @return TrackNet model input height
     */
    public int getInputHeight() {
        return inputShape[1];
    }

    /**
     *
     * @return TrackNet model output width
     */
    public int getOutputWidth() {
        return outputShape[2];
    }

    /**
     *
     * @return TrackNet model output height
     */
    public int getOutputHeight() {
        return outputShape[1];
    }

    /**
     * Preprocesses a list of Bitmap frames (length of 3) to prepare them for model inference.
     * This includes resizing, color space conversion, normalization, and channel merging.
     *
     * @param bitmaps the list of Bitmap frames to preprocess.
     * @return A ByteBuffer containing the preprocessed input data ready for inference.
     */
    public ByteBuffer preprocess(List<Bitmap> bitmaps) {

        List<Mat> concatList = new ArrayList<>();
        for (Bitmap bitmap: bitmaps) {
            // Convert Bitmap to OpenCV Mat (ABGR format)
            Mat inputMatAbgr = new Mat();
            Utils.bitmapToMat(bitmap, inputMatAbgr);

            // Resize the image to match the model's input dimensions
            Mat scaledImage = new Mat(this.getInputHeight(), this.getInputWidth(), CvType.CV_8UC4);
            Size targetSize = new Size(this.getInputWidth(), this.getInputHeight());
            Imgproc.resize(inputMatAbgr, scaledImage, targetSize, 0, 0, Imgproc.INTER_LINEAR);

//            // Convert to grayscale image
//            Mat inputMatGray = new Mat();
//            Imgproc.cvtColor(scaledImage, inputMatGray, Imgproc.COLOR_BGR2GRAY);
//            // Normalize the grayscale image to [0, 1] range
//            inputMatGray.convertTo(inputMatGray, CvType.CV_32FC1, 1/255f);
//            // If the model expects quantized inputs (UINT8), apply scaling and zero-point adjustment
//            if (inputType == DataType.UINT8) {
//                Core.divide(inputMatGray, new Scalar(INPUT_SCALE), inputMatGray);
//                Core.add(inputMatGray, new Scalar(INPUT_ZERO_POINT), inputMatGray);
//            }
//            concatList.add(inputMatGray);

            Mat inputMatRgb = new Mat();
            Imgproc.cvtColor(scaledImage, inputMatRgb, Imgproc.COLOR_BGRA2RGB);
            inputMatRgb.convertTo(inputMatRgb, CvType.CV_32FC3, 1/255f);
            if (inputType == DataType.UINT8) {
                Core.divide(inputMatRgb, INPUT_SCALE_SCALAR, inputMatRgb);
                Core.add(inputMatRgb, INPUT_ZERO_POINT_SCALAR, inputMatRgb);
            }
            concatList.add(inputMatRgb);
        }

        // Merge all processed Mats into a single Mat with multiple channels
        Mat concatenatedImage = new Mat();
        Core.merge(concatList, concatenatedImage);

        // store the preprocessed result in inputByteBuffer
        inputByteBuffer.rewind(); // Reset the buffer's position
        if (inputType == DataType.UINT8) {
            concatenatedImage.convertTo(concatenatedImage, CvType.CV_8UC3);
            concatenatedImage.get(0, 0, inputByteArray);
            inputByteBuffer.put(inputByteArray);
        } else {
            concatenatedImage.get(0, 0, inputFloatArray);
            FloatBuffer inputFloatBuffer = inputByteBuffer.asFloatBuffer();
            inputFloatBuffer.put(inputFloatArray);
        }
        return inputByteBuffer;
    }

    /**
     * perform model inferencing with TFLite interpreter
     * @param inputBuffer The ByteBuffer containing the preprocessed input data
     * @return A TensorBuffer containing the flattened output data from the model.
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
            outputBuffer = dequantProcessor.process(outputBuffer);
        }
        return outputBuffer;
    }

    /**
     * Postprocesses the model's output TensorBuffer to extract ball positions.
     * This involves thresholding, contour detection, and selecting candidates contours.
     * @param buffer The TensorBuffer containing the model's output data.
     * @return
     */
    public List<BallPos> postprocess(TensorBuffer buffer) {
        float[] outputData = buffer.getFloatArray();
        Mat outputMat = new Mat(getOutputHeight(), getOutputWidth(), CvType.CV_32FC3);
        outputMat.put(0, 0, outputData);

        // Split the output Mat into separate heatmap channels (one per ball detection heatmap)
        List<Mat> splitMatList = new ArrayList<>();
        Core.split(outputMat, splitMatList);

        List<BallPos> res = new ArrayList<>();
        // Iterate through each heatmap to detect ball positions
        for (Mat splitMat: splitMatList) {
            Mat heatmap = new Mat();
            // Apply thresholding to convert the heatmap into a binary image
            Imgproc.threshold(splitMat, heatmap, /*thresh=*/0.5, /*maxval=*/255.0, Imgproc.THRESH_BINARY);
            heatmap.convertTo(heatmap, CvType.CV_8UC1);

//            // Set the kernel (structuring element)
//            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
//
//            // Apply erosion (morphological open operation)
//            Mat maskEroded = new Mat();
//            Imgproc.morphologyEx(heatmap, maskEroded, Imgproc.MORPH_OPEN, kernel);
            // Find contours in the binary heatmap
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(heatmap, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            // Imgproc.findContours(maskEroded, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            if (!contours.isEmpty()) {
                double maxArea = 0;
                Rect boundingRect = null;
                // Iterate through all detected contours to find the largest one
                for (MatOfPoint contour : contours) {
                    Rect rect = Imgproc.boundingRect(contour);
                    double area = rect.width * rect.height;
                    if (area > maxArea) {
                        maxArea = area;
                        boundingRect = rect;
                    }
                }

                if (boundingRect != null) {
                    int x_pred = (int) (boundingRect.x + boundingRect.width / 2.0);
                    int y_pred = (int) (boundingRect.y + boundingRect.height / 2.0);
                    res.add(new BallPos(x_pred, y_pred));
                } else {
                    res.add(null);
                }
            } else {
                res.add(null);
            }
        }
        return res;
    }

}
