package com.example.activevision;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import com.example.activevision.data.BallPos;
import com.example.activevision.data.Bbox;
import com.example.activevision.result.FrameRes;
import com.example.activevision.result.TrackerResListener;
import com.example.activevision.threadings.NamingThreadFactory;
import com.example.activevision.threadings.PreprocessThreadPool;
import com.example.activevision.trackers.BallTracker;
import com.example.activevision.trackers.PlayerDetector;

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
    private final PlayerDetector playerDetector;

    // Listener to dispatch results and performance metrics
    private final TrackerResListener listener;

    // Queue to store three consecutive frames for ball tracking
    private final Queue<Bitmap> frameBuffer = new ConcurrentLinkedQueue<>();
    private final ReentrantLock mLock = new ReentrantLock(); // Lock to synchronize frame buffer operations

    // Map to store partial or complete results for each frame
    private final ConcurrentHashMap<Long, FrameRes> resultsMap = new ConcurrentHashMap<>();

    private final ThreadPoolExecutor tennisExecutor;
    private final ThreadPoolExecutor playerExecutor;

    // Frame counters
    private final AtomicLong frameCounter = new AtomicLong(1); // Global frame index for tracking
    private final AtomicLong nextFrameToRetrieve = new AtomicLong(1); // Next frame index to retrieve results

    // Performance monitoring variables
    private volatile long completedFpsCnt = 0; // FPS counter
    private final AtomicLong completedFramesCnt = new AtomicLong(0); // Number of completed frames in the last second
    private long lastTic = System.currentTimeMillis(); // Timestamp for FPS calculation


    private final int skipDetIdx = 2;

    private volatile List<Bbox> lastPlayerBboxes = null;


    public FrameAnalyzer(BallTracker tennisTracker,
                         PlayerDetector playerDetector,
                         TrackerResListener listener) {
        this.tennisTracker = tennisTracker;
        this.playerDetector = playerDetector;
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

        playerExecutor = new ThreadPoolExecutor(
                1, 1, 0L, TimeUnit.MILLISECONDS,
                // new LinkedBlockingQueue<Runnable>(),
                new PriorityBlockingQueue<>(
                        128,
                        Comparator.comparingLong(task -> ((PrioritizedTask) task).getFrameIndex())
                ),
                new NamingThreadFactory("PlayerDetThread")
        );

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

        // or skip detection in every {skipDetIdx-1} frame(s) to enhance performance
        if (frameIdx % skipDetIdx == 1) {
            PreprocessThreadPool.getInstance().submitTask(
                    new PlayerDetTask(frameIdx, input, frameHeight, frameWidth)
            );
        } else {
            playerSkipDetection(frameIdx);
        }

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
     * Skips the player detection/pose estimation task
     * This method is called when a frame is skipped from player detection.
     *
     * @param frameIdx The index of the current frame
     */
    private void playerSkipDetection(long frameIdx) {
        // If there are no player bounding boxes in the previous detection
        // then pose estimation must also be null
        if (lastPlayerBboxes == null) {
            FrameRes frameRes = resultsMap.get(frameIdx);
            if (frameRes != null) {
                frameRes.setPlayerDetList(null);
            }
            return;
        }

        // Otherwise, reuse them
        FrameRes res = resultsMap.get(frameIdx);
        if (res != null) {
            res.setPlayerDetList(lastPlayerBboxes);
        }
        retrieveFrameResult();

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
            listener.onPlayerDetCallback(res.getPlayerDetList());

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

    public class PlayerDetTask implements Runnable {
        private final long frameIdx;
        private final Bitmap inputBitmap;
        private final int inputHeight;
        private final int inputWidth;

        public PlayerDetTask(long frameIdx, Bitmap inputBitmap, int inputHeight, int inputWidth) {
            this.frameIdx = frameIdx;
            this.inputBitmap = inputBitmap;
            this.inputHeight = inputHeight;
            this.inputWidth = inputWidth;
        }

        @Override
        public void run() {
            try {
                ByteBuffer playerDetInput = playerDetector.preprocess(inputBitmap);

                playerExecutor.submit(new PrioritizedTask(frameIdx, () -> {
                    TensorBuffer outputBuffer = playerDetector.inference(playerDetInput);
                    List<Bbox> bboxes = playerDetector.postprocess(outputBuffer.getFloatArray(), inputHeight, inputWidth);
                    FrameRes res = resultsMap.get(frameIdx);
                    if (res != null) {
                        res.setPlayerDetList(bboxes);
                        lastPlayerBboxes = bboxes;
                    }
                    // Trigger pose estimation if players were detected
//                    if (bboxes != null && !bboxes.isEmpty()) {
//                        submitPoseEstimationTask(frameIdx, inputBitmap, bboxes);
//                    } else {
//                        if (res != null) {
//                            res.setFrameKps(null);
//                            lastKeypoints = null;
//                            retrieveFrameResult();
//                        }
//                    }

                }));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        private void submitPoseEstimationTask(long frameIdx, Bitmap inputBitmap, List<Bbox> bboxes) {
           //TODO: TO be implement
        }
    }
}
