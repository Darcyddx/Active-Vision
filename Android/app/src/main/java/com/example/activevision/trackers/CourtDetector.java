package com.example.activevision.trackers;

import android.content.Context;
import android.graphics.Bitmap;

import java.io.IOException;
import java.io.InputStream;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.extensions.OrtxPackage;

public class CourtDetector  implements AutoCloseable {

    private static final String TAG = "CourtDetector";

    private OrtEnvironment ortEnv;
    private OrtSession ortSession;

    public CourtDetector(Context context,
                         String modelPath) throws IOException, NoSuchAlgorithmException {

        // Initialize ONNX Runtime environment
        this.ortEnv = OrtEnvironment.getEnvironment();

        try {
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
            this.ortSession = this.ortEnv.createSession(readModel(context, modelPath), sessionOptions);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public OrtSession getSession() {
        return ortSession;
    }

    public byte[] readModel(Context context, String modelPath) {

        try {

            InputStream is = context.getAssets().open(modelPath);
            byte[] model = new byte[is.available()];
            is.read(model);
            is.close();
            return model;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new byte[0];
    }

    public float[][][][] preprocess(Bitmap image) {
        int targetSize = 640;

        // Resize the bitmap to 640x640
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(image, targetSize, targetSize, true);

        // Prepare float array in shape [1, 3, 640, 640]
        float[][][][] output = new float[1][3][targetSize][targetSize];

        // Get all pixels
        int[] pixels = new int[targetSize * targetSize];
        resizedBitmap.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize);

        // Fill the float tensor
        for (int y = 0; y < targetSize; y++) {
            for (int x = 0; x < targetSize; x++) {
                int pixel = pixels[y * targetSize + x];

                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;

                // Normalize to [0, 1]
                output[0][0][y][x] = r; // Red
                output[0][1][y][x] = g; // Green
                output[0][2][y][x] = b; // Blue
            }
        }

        return output;
    }

    public OrtSession.Result inference(float[][][][] inputTensorData) {
        OnnxTensor inputTensor = null;
        try {
            inputTensor = OnnxTensor.createTensor(this.ortEnv, inputTensorData);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }


        // run inference
        OrtSession.Result output = null;
        try {
            output = this.ortSession.run(
                    Collections.singletonMap("image", inputTensor),
                    Set.of("output", "value.19", "value.27", "3880", "value.15", "onnx::Split_3301")
            );
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }

        inputTensor.close();

        return output;
    }

    public float[][][] postprocess(OrtSession.Result result) {

        float[][][] keypoints = null; // Keypoints
        try {
            keypoints = ((float[][][]) result.get(4).getValue());
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }

        return keypoints;
    }

//    private float[] preprocessImageFloatCHW(Bitmap resizedBitmap) {
//        int width = resizedBitmap.getWidth();   // 640
//        int height = resizedBitmap.getHeight(); // 640
//
//        int[] pixels = new int[width * height];
//        resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height);
//
//        float[] input = new float[3 * width * height]; // [C, H, W]
//
//        for (int i = 0; i < pixels.length; i++) {
//            int pixel = pixels[i];
//
//            float r = (pixel >> 16) & 0xFF;
//            float g = (pixel >> 8) & 0xFF;
//            float b = pixel & 0xFF;
//
//            // normalization
////            r = r / 255.0f;
////            g = g / 255.0f;
////            b = b / 255.0f;
//
//
//            input[i] = r;                          // Red channel
//            input[i + width * height] = g;         // Green channel
//            input[i + 2 * width * height] = b;     // Blue channel
//        }
//
//        return input;
//    }

    @Override
    public void close() throws Exception {
        if (ortSession != null) ortSession.close();
        if (ortEnv != null) ortEnv.close();
    }
}