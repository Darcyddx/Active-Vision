package com.example.activevision;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import androidx.test.platform.app.InstrumentationRegistry;

import com.example.activevision.data.Bbox;
import com.example.activevision.tflite_helpers.AIHubDefaults;
import com.example.activevision.trackers.CourtDetector;

import org.junit.Before;
import org.junit.Test;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.List;

public class CourtDetTest {
    private CourtDetector courtDetector;

    private static final String IMAGE_PATH_1 = "frame_1.png";

    @Before
    public void setUp() throws Exception {
//        if (OpenCVLoader.initLocal()) {
//            Log.i("TennisTrackerTest", "OpenCV successfully loaded.");
//        }
        // Initialize the BallTracker instance with the context
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        String courtDetModelPath = context.getString(R.string.CourtDetModelAssetFP16);
        courtDetector = new CourtDetector(context, courtDetModelPath, AIHubDefaults.delegatePriorityOrder);
        System.out.println("finish set up");
    }

    @Test
    public void testCourtDetPreprocess() throws Exception {
        // Context to access assets
        Context context = InstrumentationRegistry.getInstrumentation().getContext();

        // Load sample PNG images from assets
        Bitmap bitmap1 = loadBitmapFromAssets(context, IMAGE_PATH_1);

        assert bitmap1 != null;
        ByteBuffer output = courtDetector.preprocess(bitmap1);

        System.out.println("complete pre-processing");
    }

    @Test
    public void testCourtDetModel() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getContext();

        // Load sample PNG images from assets
        Bitmap bitmap1 = loadBitmapFromAssets(context, IMAGE_PATH_1);

        assert bitmap1 != null;
        ByteBuffer inputByteBuffer = courtDetector.preprocess(bitmap1);

        TensorBuffer outputBuffer = courtDetector.inference(inputByteBuffer);
        float[] inf = outputBuffer.getFloatArray();
        List<float[]> res = courtDetector.postprocess(inf,bitmap1.getHeight(), bitmap1.getWidth());
//        float[][] res = courtDetector.processOutput(outputBuffer);

        // Create a mutable copy of the bitmap to draw on it
        Bitmap mutableBitmap = bitmap1.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);

        // Set up the paint to draw keypionts and lines
        Paint courtkpPaint = new Paint();
        courtkpPaint.setColor(Color.GREEN);
        courtkpPaint.setStyle(Paint.Style.FILL);

        Paint courtLinePaint = new Paint();
        courtLinePaint.setColor(Color.GREEN);
        courtLinePaint.setStyle(Paint.Style.STROKE);
        courtLinePaint.setStrokeWidth(4f);

        int[][] connections = {
                {0, 1}, {0, 10}, {1, 2}, {1, 4},
                {2, 3}, {2, 6}, {3, 13},
                {4, 5}, {4, 7}, {5, 6}, {5, 8}, {6, 9},
                {7, 11}, {7, 8}, {8, 9}, {9, 12}, {10, 11},
                {11, 12}, {12, 13}
        };


        // Draw court keypoints
        for (int i = 0; i < 14; i++) {
            float[] kp = res.get(i);
            if (kp != null && kp.length == 2 && kp[0] >= 0 && kp[1] >= 0) {
                canvas.drawCircle(kp[0], kp[1], 6f, courtkpPaint);
            }
        }

        // Draw court lines
        for (int[] line : connections) {
            int i1 = line[0];
            int i2 = line[1];

            if (i1 < res.size() && i2 < res.size()) {
                float[] kp1 = res.get(i1);
                float[] kp2 = res.get(i2);

                if (kp1 != null && kp2 != null &&
                        kp1.length == 2 && kp2.length == 2 &&
                        kp1[0] >= 0 && kp1[1] >= 0 && kp2[0] >= 0 && kp2[1] >= 0) {

                    canvas.drawLine(kp1[0], kp1[1], kp2[0], kp2[1], courtLinePaint);
                }
            }
        }

        System.out.println("done");

    }


    private Bitmap loadBitmapFromAssets(Context context, String fileName) {
        try {
            InputStream inputStream = context.getAssets().open(fileName);
            return BitmapFactory.decodeStream(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
