package com.example.activevision;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import com.example.activevision.data.BallPos;
import com.example.activevision.result.FrameRes;
import com.example.activevision.result.TrackerResListener;
import com.example.activevision.threadings.NamingThreadFactory;
import com.example.activevision.threadings.PreprocessThreadPool;
import com.example.activevision.trackers.BallTracker;

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

public class FrameAnalyzer implements ImageAnalysis.Analyzer {

    private static final String TAG = "FrameAnalyzer";

    private final BallTracker tennisTracker;

    // Listener to dispatch results and performance metrics
    private final TrackerResListener listener;

    // Queue to store three consecutive frames for ball tracking
    private final Queue<Bitmap> frameBuffer = new ConcurrentLinkedQueue<>();
    private final ReentrantLock mLock = new ReentrantLock(); // Lock to synchronize frame buffer operations

    // Map to store partial or complete results for each frame
    private final ConcurrentHashMap<Long, FrameRes> resultsMap = new ConcurrentHashMap<>();

    private final ThreadPoolExecutor tennisExecutor;

    // Frame counters
    private final AtomicLong frameCounter = new AtomicLong(1); // Global frame index for tracking
    private final AtomicLong nextFrameToRetrieve = new AtomicLong(1); // Next frame index to retrieve results

    // Performance monitoring variables
    private volatile long completedFpsCnt = 0; // FPS counter
    private final AtomicLong completedFramesCnt = new AtomicLong(0); // Number of completed frames in the last second
    private long lastTic = System.currentTimeMillis(); // Timestamp for FPS calculation


    private final int skipDetIdx = 2;


    public FrameAnalyzer(BallTracker tennisTracker,
                         TrackerResListener listener) {
        this.tennisTracker = tennisTracker;

        this.listener = listener;

        tennisExecutor = new ThreadPoolExecutor(
                1,
                1,
                0L,
                TimeUnit.MILLISECONDS,
                new PriorityBlockingQueue<>(128,
                        Comparator.comparingLong(task -> ((PrioritizedTask) task).getFrameIndex())
                ),
                new NamingThreadFactory("BallTrackThread"));

    }
    @Override
    public void analyze(@NonNull ImageProxy image) {
        int frameWidth = image.getWidth();
        int frameHeight = image.getHeight();

        Bitmap input = image.toBitmap();

        // Assign a unique frame index to submit into priority task queue later
        long frameIdx = frameCounter.getAndIncrement();

        // to store results for current frame
        FrameRes saveRes = new FrameRes(frameIdx);
        resultsMap.put(frameIdx, saveRes);

        // Perform ball tracking (requires 3 consecutive frames)
        mLock.lock();
        try {
            frameBuffer.add(input);
            if (frameBuffer.size() == 3) {
                List<Bitmap> frames = new ArrayList<>(frameBuffer);
                frameBuffer.clear();
                // Once 3 frames are collected, process them for ball tracking
                PreprocessThreadPool.getInstance().submitTask(
                        new BallTrackingTask(frameIdx, frames)
                );
            }
        } finally {
            mLock.unlock();
        }

        image.close();
    }

    /**
     * Retrieves completed results from the results map in sequential order, starting from the next frame to retrieve.
     * If a frame is complete, removes it and callback results.
     */
    private void retrieveFrameResult() {
        while (true) {
            long retrieveIdx = nextFrameToRetrieve.get();
            FrameRes res = resultsMap.get(retrieveIdx);
            if (res == null || !res.isComplete()) {
                // Stop if no result or result is incomplete
                break;
            }

            // Remove the completed frame
            resultsMap.remove(retrieveIdx);
            nextFrameToRetrieve.incrementAndGet();

            listener.onBallPosCallback(res.getBallPositions());


            // Calculate FPS
            completedFramesCnt.incrementAndGet();
            long currentTic = System.currentTimeMillis();
            if (currentTic - lastTic >= 1000) {
                completedFpsCnt = completedFramesCnt.get();
                completedFramesCnt.set(0);
                lastTic = currentTic;
                Log.d("FrameAnalyzer", "Completed FPS: " + completedFpsCnt);
                listener.onPerformanceCallback(completedFpsCnt);
            }
        }
    }

    /**
     * Shuts down all threads and executors used by the pipeline.
     */
    public void shutdownExecutor() {
        PreprocessThreadPool.getInstance().shutdown();

        tennisExecutor.shutdownNow();

    }


    /**
     * A prioritized task to submit into model inference thread pool.
     * It is used to prioritize tasks based on their frame index.
     */
    public class PrioritizedTask implements Runnable, Comparable<PrioritizedTask> {
        private final long frameIndex; // The frame index for prioritization
        private final Runnable task;   // The actual task logic to execute

        public PrioritizedTask(long frameIndex, Runnable task) {
            this.frameIndex = frameIndex;
            this.task = task;
        }

        public long getFrameIndex() {
            return frameIndex;
        }

        @Override
        public void run() {
            task.run();
        }

        @Override
        public int compareTo(PrioritizedTask other) {
            return Long.compare(this.frameIndex, other.frameIndex);
        }
    }

    private class BallTrackingTask implements Runnable {
        private final long frameIdx;
        private final List<Bitmap> frames;

        public BallTrackingTask(long frameIdx, List<Bitmap> frames) {
            this.frameIdx = frameIdx;
            this.frames = frames;
        }

        @Override
        public void run() {
            try {
                // Preprocess frames for ball tracking
                ByteBuffer ballInput = tennisTracker.preprocess(frames);

                tennisExecutor.submit(new PrioritizedTask(frameIdx, () -> {

                    TensorBuffer outputBuffer = tennisTracker.inference(ballInput);
                    List<BallPos> ballPosList = tennisTracker.postprocess(outputBuffer);

                    FrameRes res = resultsMap.get(frameIdx);
                    if (res != null) {
                        res.setBallPosList(ballPosList);
                    }
                    retrieveFrameResult();
                }));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
