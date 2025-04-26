package com.example.active_vision_qualcomm;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import androidx.test.platform.app.InstrumentationRegistry;

import com.example.active_vision_qualcomm.Trackers.PlayerDetector;
import com.example.active_vision_qualcomm.Trackers.PlayerPoseTracker;
import com.example.active_vision_qualcomm.data.Bbox;
import com.example.active_vision_qualcomm.data.KeyPoint;
import com.example.active_vision_qualcomm.data.PoseInferenceInfo;
import com.example.active_vision_qualcomm.data.PosePreprocessInfo;
import com.example.active_vision_qualcomm.tflite_helpers.AIHubDefaults;

import org.junit.Before;
import org.junit.Test;
import org.opencv.core.Point;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class PoseEstimationTest {
    private PlayerDetector playerDetector;

    private PlayerPoseTracker playerPoseTracker;

    private static final String IMAGE_PATH_1 = "player.jpg";

    @Before
    public void setUp() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        String playerDetModelPath = context.getString(R.string.PlayerDetModelAssetUINT8);
        playerDetector = new PlayerDetector(context, playerDetModelPath, AIHubDefaults.delegatePriorityOrder);

        String poseEstModelPath = context.getString(R.string.PoseEstMobileModelAssetFP16);
        playerPoseTracker = new PlayerPoseTracker(context, poseEstModelPath, AIHubDefaults.delegatePriorityOrder);
    }

    @Test
    public void testPoseEstModel() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getContext();

        Bitmap bitmap1 = loadBitmapFromAssets(context, IMAGE_PATH_1);

        assert bitmap1 != null;
        ByteBuffer inputByteBuffer = playerDetector.preprocess(bitmap1);

        TensorBuffer outputBuffer = playerDetector.inference(inputByteBuffer);
        float[] inf = outputBuffer.getFloatArray();
        List<Bbox> bboxes = playerDetector.postprocess(inf, bitmap1.getHeight(), bitmap1.getWidth());

        List<PosePreprocessInfo> inputs = playerPoseTracker.preprocess(bitmap1, bboxes);

        List<PoseInferenceInfo> outputs = playerPoseTracker.inference(inputs);


        List<List<KeyPoint>> frameKps = playerPoseTracker.postprocess(outputs);

        Bitmap mutableBitmap = bitmap1.copy(Bitmap.Config.ARGB_8888, true);

        // Create a canvas from the mutable bitmap.
        Canvas canvas = new Canvas(mutableBitmap);

        // Prepare a Paint object.
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(Color.RED);         // Example color: red
        paint.setStyle(Paint.Style.FILL);  // Filled circle
        paint.setStrokeWidth(5f);          // Thickness if needed

        float radius = 3f;
        for (int i = 0; i < frameKps.size(); i++) {
            List<KeyPoint> playerKps = frameKps.get(i);
            for (KeyPoint kp: playerKps) {
                float x = (float) kp.getPoint().x;
                float y = (float) kp.getPoint().y;

                // Draw the circle on the canvas.
                canvas.drawCircle(x, y, radius, paint);
            }
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
