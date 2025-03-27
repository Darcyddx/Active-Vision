package com.example.activevision;

import static org.junit.Assert.assertNotNull;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Pair;

import androidx.test.platform.app.InstrumentationRegistry;

import com.example.activevision.data.BallPos;
import com.example.activevision.tflite_helpers.AIHubDefaults;
import com.example.activevision.tflite_helpers.TFLiteHelpers;
import com.example.activevision.trackers.BallTracker;

import org.junit.Before;
import org.junit.Test;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class TennisTrackerTest {
    private BallTracker tennisTracker;
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
        String trackNetPath = context.getString(R.string.TrackNetModelAssetUINT8);
        tennisTracker = new BallTracker(context, trackNetPath, AIHubDefaults.delegatePriorityOrder);
        System.out.println("finish set up");
    }

    @Test
    public void testLoadModel() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Interpreter tfLiteInterpreter;
        Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;

        String modelPath = context.getString(R.string.TrackNetModelAssetUINT8);
        Pair<MappedByteBuffer, String> modelAndHash = TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);
        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                modelAndHash.first,
                AIHubDefaults.delegatePriorityOrder,
                AIHubDefaults.numCPUThreads,
                context.getApplicationInfo().nativeLibraryDir,
                context.getCacheDir().getAbsolutePath(),
                modelAndHash.second
        );
        tfLiteInterpreter = iResult.first;
        tfLiteDelegateStore = iResult.second;
        assert tfLiteInterpreter.getInputTensorCount() == 1;
        Tensor inputTensor = tfLiteInterpreter.getInputTensor(0);

    }

    @Test
    public void testPreprocess() throws Exception {
        System.out.println("test pre-processing");
    }

    @Test
    public void testTrackNetInference() throws Exception {
        // Context to access assets
        Context context = InstrumentationRegistry.getInstrumentation().getContext();

        // Load sample PNG images from assets
        Bitmap bitmap1 = loadBitmapFromAssets(context, IMAGE_PATH_1);
        Bitmap bitmap2 = loadBitmapFromAssets(context, IMAGE_PATH_2);
        Bitmap bitmap3 = loadBitmapFromAssets(context, IMAGE_PATH_3);

        // Create a list of bitmaps to pass to the inference method
        List<Bitmap> bitmaps = Arrays.asList(bitmap1, bitmap2, bitmap3);
        long preprocessStartTime = System.nanoTime();
        //TensorBuffer inputs = tennisTracker.preprocess(bitmaps);
        ByteBuffer inputs = tennisTracker.preprocess(bitmaps);
        long preprocessEndTime = System.nanoTime();
        long preprocessDuration = preprocessEndTime - preprocessStartTime;
        System.out.println("Preprocessing time: " + preprocessDuration / 1_000_000.0 + " ms");

        // Call the inference method
        long inferenceStartTime = System.nanoTime();
        TensorBuffer outputBuffer = tennisTracker.inference(inputs);
        float[] inf = outputBuffer.getFloatArray();
        long inferenceEndTime = System.nanoTime();
        long inferenceDuration = inferenceEndTime - inferenceStartTime;
        System.out.println("Inference time: " + inferenceDuration / 1_000_000.0 + " ms");

        // Check that the output tensor is not null
        assertNotNull(outputBuffer);

        long postProcessStartTime = System.nanoTime();
        List<BallPos> ret = tennisTracker.postprocess(outputBuffer);
        long postProcessEndTime = System.nanoTime();
        long postProcessDuration = postProcessEndTime - postProcessStartTime;
        System.out.println("Post process time: " + postProcessDuration / 1_000_000.0 + " ms");
        for (int i = 0; i < ret.size(); i++) {
            System.out.println("get ball pos predicted: " + ret.get(i));
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
