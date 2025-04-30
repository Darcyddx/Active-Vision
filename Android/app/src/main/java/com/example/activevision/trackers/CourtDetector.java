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
        ortEnv = OrtEnvironment.getEnvironment();

        // Session options (register custom ops if needed)
//        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

        try {
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
            ortSession = ortEnv.createSession(readModel(context, modelPath), sessionOptions);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public OrtSession getSession() {
        return ortSession;
    }

    private byte[] readModel(Context context, String modelPath) {

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
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(image, 640, 640, true); // Resize as per your model

        float[] inputData = preprocessImageFloatCHW(resizedBitmap);
        float[][][][] inputTensorData = new float[1][3][640][640];

        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 640; h++) {
                for (int w = 0; w < 640; w++) {
                    inputTensorData[0][c][h][w] = inputData[c * 640 * 640 + h * 640 + w];
                }
            }
        }

        return inputTensorData;
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

    private float[] preprocessImageFloatCHW(Bitmap resizedBitmap) {
        int width = resizedBitmap.getWidth();   // 640
        int height = resizedBitmap.getHeight(); // 640

        int[] pixels = new int[width * height];
        resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] input = new float[3 * width * height]; // [C, H, W]

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];

            float r = (pixel >> 16) & 0xFF;
            float g = (pixel >> 8) & 0xFF;
            float b = pixel & 0xFF;

            // normalization
//            r = r / 255.0f;
//            g = g / 255.0f;
//            b = b / 255.0f;


            input[i] = r;                          // Red channel
            input[i + width * height] = g;         // Green channel
            input[i + 2 * width * height] = b;     // Blue channel
        }

        return input;
    }

    @Override
    public void close() throws Exception {
        if (ortSession != null) ortSession.close();
        if (ortEnv != null) ortEnv.close();
    }
}