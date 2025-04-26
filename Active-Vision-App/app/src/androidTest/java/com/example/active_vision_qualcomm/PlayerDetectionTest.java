package com.example.active_vision_qualcomm;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import androidx.test.platform.app.InstrumentationRegistry;

import com.example.active_vision_qualcomm.Trackers.PlayerDetector;
import com.example.active_vision_qualcomm.Trackers.TennisTracker;
import com.example.active_vision_qualcomm.data.Bbox;
import com.example.active_vision_qualcomm.tflite_helpers.AIHubDefaults;

import org.junit.Before;
import org.junit.Test;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.List;

public class PlayerDetectionTest {
    private PlayerDetector playerDetector;

    private static final String IMAGE_PATH_1 = "player.jpg";

    @Before
    public void setUp() throws Exception {
//        if (OpenCVLoader.initLocal()) {
//            Log.i("TennisTrackerTest", "OpenCV successfully loaded.");
//        }
        // Initialize the BallTracker instance with the context
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        String playerDetModelPath = context.getString(R.string.PlayerDetModelAssetUINT8);
        playerDetector = new PlayerDetector(context, playerDetModelPath, AIHubDefaults.delegatePriorityOrder);
        System.out.println("finish set up");
    }

    @Test
    public void testPlayerDetPreprocess() throws Exception {
        // Context to access assets
        Context context = InstrumentationRegistry.getInstrumentation().getContext();

        // Load sample PNG images from assets
        Bitmap bitmap1 = loadBitmapFromAssets(context, IMAGE_PATH_1);

        assert bitmap1 != null;
        ByteBuffer output = playerDetector.preprocess(bitmap1);

        System.out.println("complete pre-processing");
    }

    @Test
    public void testPlayerDetModel() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getContext();

        // Load sample PNG images from assets
        Bitmap bitmap1 = loadBitmapFromAssets(context, IMAGE_PATH_1);

        assert bitmap1 != null;
        ByteBuffer inputByteBuffer = playerDetector.preprocess(bitmap1);

        TensorBuffer outputBuffer = playerDetector.inference(inputByteBuffer);
        float[] inf = outputBuffer.getFloatArray();
        List<Bbox> res = playerDetector.postprocess(inf, bitmap1.getHeight(), bitmap1.getWidth());

        // Create a mutable copy of the bitmap to draw on it
        Bitmap mutableBitmap = bitmap1.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);

        // Set up the paint for drawing bounding boxes
        Paint paint = new Paint();
        paint.setColor(Color.RED); // Set bounding box color
        paint.setStyle(Paint.Style.STROKE); // Outline only
        paint.setStrokeWidth(2f); // Thickness of the bounding box


        for (Bbox box: res) {
            float left = box.getRect().left * bitmap1.getWidth();
            float top = box.getRect().top * bitmap1.getWidth();
            float right = box.getRect().right * bitmap1.getWidth();
            float bottom = box.getRect().bottom * bitmap1.getWidth();
            // Draw the bounding box
            canvas.drawRect(left, top, right, bottom, paint);
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
