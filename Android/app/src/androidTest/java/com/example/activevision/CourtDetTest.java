package com.example.activevision;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import androidx.test.platform.app.InstrumentationRegistry;

import com.example.activevision.trackers.CourtDetector;

import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.extensions.OrtxPackage;

public class CourtDetTest {

    private CourtDetector courtDetector;

    private OrtEnvironment ortEnv;
    private OrtSession ortSession;

    private static final String IMAGE_PATH_1 = "frame_1.png";
    private static final String IMAGE_PATH_2 = "frame_2.png";
    private static final String IMAGE_PATH_3 = "frame_3.png";

    @Before
    public void setUp() throws Exception {
//        if (OpenCVLoader.initLocal()) {
//            Log.i("TennisTrackerTest", "OpenCV successfully loaded.");
//        }
        // Initialize the BallTracker instance with the context
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        String courtDetPath = context.getString(R.string.CourtDetectionModelAssetOnnx);
        courtDetector = new CourtDetector(context, courtDetPath);
        System.out.println("finish set up");
    }

    @Test
    public void testLoadModel() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();

        String modelPath = context.getString(R.string.CourtDetectionModelAssetOnnx);

        ortEnv = OrtEnvironment.getEnvironment();

        try {
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
            ortSession = ortEnv.createSession(courtDetector.readModel(context, modelPath), sessionOptions);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Test
    public void testPreprocess() throws Exception {
        System.out.println("test pre-processing");
    }

    @Test
    public void testCourtDetInference() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getContext();

        // Load sample PNG images from assets
        Bitmap bitmap1 = loadBitmapFromAssets(context, IMAGE_PATH_1);

        assert bitmap1 != null;
        float[][][][] inputByteBuffer = courtDetector.preprocess(bitmap1);

        OrtSession.Result outputBuffer = courtDetector.inference(inputByteBuffer);
        float[][][] res = courtDetector.postprocess(outputBuffer);

        // Create a mutable copy of the bitmap to draw on it
        Bitmap mutableBitmap = bitmap1.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);

        int origWidth = mutableBitmap.getWidth();
        int origHeight = mutableBitmap.getHeight();

        float scaleX = (float) origWidth / 640f;
        float scaleY = (float) origHeight / 640f;

        Paint kpPaint = new Paint();
        kpPaint.setColor(Color.GREEN);
        kpPaint.setStyle(Paint.Style.FILL);

        Paint linePaint = new Paint();
        linePaint.setColor(Color.GREEN);
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(4f);

        float[][] keypoints = res[0]; // [14][3]
        float[][] scaledPoints = new float[14][2]; // Scaled to original size

        // Draw keypoints & record scaled positions
        for (int i = 0; i < 14; i++) {
            float x = keypoints[i][0] * scaleX;
            float y = keypoints[i][1] * scaleY;
            float conf = keypoints[i][2];

            if (conf > 0.0f) {
                scaledPoints[i][0] = x;
                scaledPoints[i][1] = y;
                canvas.drawCircle(x, y, 6f, kpPaint);
            } else {
                scaledPoints[i][0] = -1;
                scaledPoints[i][1] = -1;
            }
        }

        // Define the keypoint connection lines
        int[][] lines = {
                {0, 1}, {0, 10}, {1, 2}, {1, 4},
                {2, 3}, {2, 6}, {3, 13},
                {4, 5}, {4, 7}, {5, 6}, {5, 8}, {6, 9},
                {7, 11}, {7, 8}, {8, 9}, {9, 12}, {10, 11},
                {11, 12}, {12, 13}
        };

        // Draw connecting lines
        for (int[] line : lines) {
            int i1 = line[0];
            int i2 = line[1];
            float x1 = scaledPoints[i1][0];
            float y1 = scaledPoints[i1][1];
            float x2 = scaledPoints[i2][0];
            float y2 = scaledPoints[i2][1];

            if (x1 >= 0 && x2 >= 0 && y1 >= 0 && y2 >= 0) {
                canvas.drawLine(x1, y1, x2, y2, linePaint);
            }

            System.out.println("done");
        }
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
