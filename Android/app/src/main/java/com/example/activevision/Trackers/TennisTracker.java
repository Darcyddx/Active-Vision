package com.example.activevision.Trackers;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Pair;

import com.example.activevision.data.BallPos;

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

public class TennisTracker implements AutoCloseable {
    private static final String TAG = "TennisTracker";

    private static final float STD = 255.0f;
    private static final float MEAN = 0.0f;

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

    public TennisTracker(Context context,
                         String modelPath)
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

            // Convert to grayscale image
            Mat inputMatGray = new Mat();
            Imgproc.cvtColor(scaledImage, inputMatGray, Imgproc.COLOR_BGR2GRAY);
            // Normalize the grayscale image to [0, 1] range
            inputMatGray.convertTo(inputMatGray, CvType.CV_32FC1, 1/255f);
            // If the model expects quantized inputs (UINT8), apply scaling and zero-point adjustment
            if (inputType == DataType.UINT8) {
                Core.divide(inputMatGray, new Scalar(INPUT_SCALE), inputMatGray);
                Core.add(inputMatGray, new Scalar(INPUT_ZERO_POINT), inputMatGray);
            }
            concatList.add(inputMatGray);
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
