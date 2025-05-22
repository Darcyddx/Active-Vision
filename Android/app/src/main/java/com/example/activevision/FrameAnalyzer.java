package com.example.activevision;

import android.graphics.Bitmap;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Pair;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import com.example.activevision.data.BallPos;
import com.example.activevision.data.Bbox;
import com.example.activevision.data.KeyPoint;
import com.example.activevision.data.PoseInferenceInfo;
import com.example.activevision.data.PosePreprocessInfo;
import com.example.activevision.data.PreprocessData;
import com.example.activevision.result.FrameRes;
import com.example.activevision.result.TrackerResListener;
import com.example.activevision.threadings.NamingThreadFactory;
import com.example.activevision.threadings.PreprocessThreadPool;
import com.example.activevision.trackers.BallTracker;
import com.example.activevision.trackers.CourtDetector;
import com.example.activevision.trackers.PlayerDetector;
import com.example.activevision.trackers.PlayerPoseEstimator;
import com.example.activevision.trackers.PlayerPoseTracker;

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;//add it if it don't have

/**
 * The FrameAnalyzer class integrates TensorFlow Lite for model inference to detect and track
 * the position of tennis balls in video frames. It leverages optimized machine learning models
 * and hardware acceleration using delegates like GPU and NPU to achieve real-time performance.
 * Author: Zhiyuan Lu, Yichi Zhang
 * Date: 21/03/2025
 */
public class FrameAnalyzer implements ImageAnalysis.Analyzer {

    private static final String TAG = "FrameAnalyzer";

    private final BallTracker tennisTracker;
    private final PlayerDetector playerDetector;

    private final PlayerPoseTracker playerPoseTracker;
    private final PlayerPoseEstimator playerPoseEstimator;

    private final CourtDetector courtDetector;

    // Listener to dispatch results and performance metrics
    private final TrackerResListener listener;

    // Queue to store three consecutive frames for ball tracking
    private final Queue<Bitmap> frameBuffer = new ConcurrentLinkedQueue<>();
    private final ReentrantLock mLock = new ReentrantLock(); // Lock to synchronize frame buffer operations

    // Queues to hold preprocessed data.
    private final PriorityBlockingQueue<PreprocessData<Pair<Bitmap, ByteBuffer>>> playerDetInputPq = new PriorityBlockingQueue<>();
    private final PriorityBlockingQueue<PreprocessData<ByteBuffer>> ballTrackInputPq = new PriorityBlockingQueue<>();

    private final PriorityBlockingQueue<PreprocessData<List<PosePreprocessInfo>>> poseEstInputPq = new PriorityBlockingQueue<>();

    private final PriorityBlockingQueue<PreprocessData<Pair<Bitmap, ByteBuffer>>> courtDetInputPq = new PriorityBlockingQueue<>();


    // Map to store partial or complete results for each frame
    private final ConcurrentHashMap<Long, FrameRes> resultsMap = new ConcurrentHashMap<>();

    // Thread pool for preprocessing frames
    //private final ThreadPoolExecutor executor;

    // Separate single-thread executors for model inferences
    private final ExecutorService playerExecutor;
    private final ExecutorService tennisExecutor;

    private final ExecutorService poseExecutor;

    private final ExecutorService courtExecutor;


    // HandlerThread for coordinating inference scheduling
    private final HandlerThread inferenceThread;
    private final Handler inferenceHandler;

    // Frame counters
    private final AtomicLong frameCounter = new AtomicLong(1); // Global frame index for tracking
    private final AtomicLong nextFrameToRetrieve = new AtomicLong(1); // Next frame index to retrieve results

    // Performance monitoring variables
    private volatile long completedFpsCnt = 0; // FPS count
    private final AtomicLong completedFramesCnt = new AtomicLong(0); // Number of completed frames in the last second
    private long lastTic = System.currentTimeMillis(); // Timestamp for FPS calculation

    private int cameraCapturedHeight;

    private int cameraCapturedWidth;

    private final int skipDetIdx = 2;

    private List<Bbox> lastPlayerBboxes = null;
    private List<List<KeyPoint>> lastKeypoints = null;
    private List<float[]> lastCourtKeypoints = null;

    /**
     * Constructor to initialize the frame analyzer pipeline.
     * @param tennisTracker TensorFlow Lite model for ball tracking.
     * @param playerDetector TensorFlow Lite model for player detection.
     * @param listener Callback to handle results and performance updates.
     */
    public FrameAnalyzer(BallTracker tennisTracker,
                         PlayerDetector playerDetector,
                         PlayerPoseTracker playerPoseTracker,
                         PlayerPoseEstimator playerPoseEstimator,
                         CourtDetector courtDetector,
                         TrackerResListener listener) {
        this.tennisTracker = tennisTracker;
        this.playerDetector = playerDetector;
        this.playerPoseTracker = playerPoseTracker;
        this.playerPoseEstimator = playerPoseEstimator;
        this.courtDetector = courtDetector;
        this.listener = listener;

        // Initialize single-thread executors for inference (one for each model)
        playerExecutor = java.util.concurrent.Executors.newSingleThreadExecutor();
        tennisExecutor = java.util.concurrent.Executors.newSingleThreadExecutor();
        poseExecutor = java.util.concurrent.Executors.newSingleThreadExecutor();
        courtExecutor = java.util.concurrent.Executors.newSingleThreadExecutor();


        // Initialize a HandlerThread for orchestrating inference tasks
        inferenceThread = new HandlerThread("InferenceCoordinatorThread");
        inferenceThread.start();
        inferenceHandler = new Handler(inferenceThread.getLooper());

    }

    public int getCameraCapturedHeight() {
        return cameraCapturedHeight;
    }

    public int getCameraCapturedWidth() {
        return cameraCapturedWidth;
    }

    /**
     * Called by CameraX for each frame to process.
     * Converts the frame to a Bitmap, preprocesses it, and schedules inferences for models.
     */
    @Override
    public void analyze(@NonNull ImageProxy image) {
        // Step 1: Convert the ImageProxy to Bitmap for further processing
        int frameWidth = image.getWidth();
        int frameHeight = image.getHeight();
        cameraCapturedHeight = frameHeight;
        cameraCapturedWidth = frameWidth;
//        Bitmap input = Bitmap.createBitmap(frameWidth, frameHeight, Bitmap.Config.ARGB_8888);
//        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
//        buffer.rewind();
//        input.copyPixelsFromBuffer(buffer);
        Bitmap input = image.toBitmap();

        // Step 2: Assign a unique frame index
        long frameIdx = frameCounter.getAndIncrement();

        // Step 3: Create an empty result container for this frame and store it in the hashmap
        FrameRes res = new FrameRes(frameIdx);
        resultsMap.put(frameIdx, res);

        if (frameIdx % skipDetIdx == 1) {
            // Step 4: Preprocess the frame for player detection and schedule inference
            PreprocessThreadPool.getInstance().submitTask(() -> {
                ByteBuffer playerDetInput = playerDetector.preprocess(input);
                playerDetInputPq.put(new PreprocessData<>(frameIdx, new Pair<>(input, playerDetInput))); // Enqueue preprocessed data
                // Schedule player detection inference
                inferenceHandler.post(() -> {
                    try {
                        runPlayerDet(frameHeight, frameWidth);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                });
            });
        } else {
            playerSkipDetection(frameIdx);
        }

         // Step 5: Preprocess the frame for court detection and schedule inference
        if (frameIdx % skipDetIdx == 1) {
            // Step 6: Preprocess the frame for court detection and schedule inference
            PreprocessThreadPool.getInstance().submitTask(() -> {
                ByteBuffer courtDetInput = courtDetector.preprocess(input);
                courtDetInputPq.put(new PreprocessData<>(frameIdx, new Pair<>(input, courtDetInput))); // Enqueue preprocessed data
                // Schedule court detection inference
                inferenceHandler.post(() -> {
                    try {
                        runCourtDet(frameHeight, frameWidth);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                });
            });
        } else {
            courtSkipDetection(frameIdx);
        }


        // Step 7: Collect 3 consecutive frames for ball tracking
        mLock.lock();
        try {
            frameBuffer.add(input);
            if (frameBuffer.size() == 3) {
                // Once 3 frames are collected, preprocess for ball tracking
                List<Bitmap> framesToProcess = new ArrayList<>(frameBuffer);
                frameBuffer.clear();

                PreprocessThreadPool.getInstance().submitTask(() -> {
                    ByteBuffer ballTrackerInput = tennisTracker.preprocess(framesToProcess);
                    ballTrackInputPq.put(new PreprocessData<>(frameIdx, ballTrackerInput)); // Enqueue preprocessed data

                    // Schedule ball tracking inference
                    inferenceHandler.post(() -> {
                        try {
                            runBallTracker();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    });
                });
            }
        } finally {
            mLock.unlock();
        }

        // Release the image to avoid memory leaks
        image.close();
    }


    /**
     * Runs ball tracking inference on the tennisExecutor and stores the results.
     * @throws InterruptedException if the thread is interrupted while waiting.
     */
    private void runBallTracker() throws InterruptedException {
        PreprocessData<ByteBuffer> processedData = ballTrackInputPq.poll(); // Retrieve the preprocessed frame if container is not empty
        if (processedData != null) {
            long idx = processedData.getFrameIndex();
            ByteBuffer inputBuffer = processedData.getData();

            // Run ball tracking inference on one thread
            tennisExecutor.submit(() -> {
                TensorBuffer outputBuffer = tennisTracker.inference(inputBuffer);
                List<BallPos> tennisPosList = tennisTracker.postprocess(outputBuffer);

                // Store the results for this frame
                FrameRes frameRes = resultsMap.get(idx);
                if (frameRes != null) {
                    frameRes.setBallPosList(tennisPosList);
                }

                // Attempt to retrieve completed results
                retrieveFrameResult();
            });
        }
    }

    /**
     * Runs player detection inference on the playerExecutor and stores the results.
     * @param inputHeight Height of the input frame.
     * @param inputWidth Width of the input frame.
     * @throws InterruptedException if the thread is interrupted while waiting.
     */
    private void runPlayerDet(int inputHeight, int inputWidth) throws InterruptedException {
        PreprocessData<Pair<Bitmap, ByteBuffer>> processedData = playerDetInputPq.poll(); // Retrieve the next preprocessed frame
        if (processedData != null) {
            long idx = processedData.getFrameIndex();
            ByteBuffer inputBuffer = processedData.getData().second;

            // Run player detection inference on one thread
            playerExecutor.submit(() -> {
                TensorBuffer outputBuffer = playerDetector.inference(inputBuffer);
                float[] inf = outputBuffer.getFloatArray();
                List<Bbox> bboxes = playerDetector.postprocess(inf, inputHeight, inputWidth);
                // Store the results for this frame
                FrameRes frameRes = resultsMap.get(idx);
                if (bboxes != null && !bboxes.isEmpty()) {
                    lastPlayerBboxes = bboxes;
                    Bitmap originBitmap = processedData.getData().first;
                    PreprocessThreadPool.getInstance().submitTask(() -> {
                        List<PosePreprocessInfo> posePreInfoList = playerPoseTracker.preprocess(originBitmap, bboxes);
                        poseEstInputPq.put(new PreprocessData<>(idx, posePreInfoList));
                        inferenceHandler.post(() -> {
                            try {
                                runPoseEst(); // define runPoseEst similarly to runBallTracker or runPlayerDet
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                        });
                    });
                } else {
                    lastKeypoints = null;
                    if (frameRes != null) {
                        frameRes.setFrameKps(null);
                    }
                }
                if (frameRes != null) {
                    lastPlayerBboxes = bboxes; // <--- store for reuse
                    frameRes.setPlayerDetList(bboxes);
                }

                // Attempt to retrieve completed results
                retrieveFrameResult();
            });
        }
    }

    private void runPoseEst() throws InterruptedException {
        PreprocessData<List<PosePreprocessInfo>> processedData = poseEstInputPq.poll();
        if (processedData != null) {
            long idx = processedData.getFrameIndex();
            List<PosePreprocessInfo> posePreList = processedData.getData();
            poseExecutor.submit(() -> {
                List<PoseInferenceInfo> outputs = playerPoseTracker.inference(posePreList);
                List<List<KeyPoint>> frameKps = playerPoseTracker.postprocess(outputs);

                // Store the results for this frame
                FrameRes frameRes = resultsMap.get(idx);
                if (frameRes != null) {
                    frameRes.setFrameKps(frameKps);
                }
                lastKeypoints = frameKps;
                // Attempt to retrieve completed results
                retrieveFrameResult();
            });
        }
    }

    private void playerSkipDetection(long frameIdx) {
        if (lastPlayerBboxes == null) {
            FrameRes frameRes = resultsMap.get(frameIdx);
            if (frameRes != null) {
                frameRes.setFrameKps(null);
                frameRes.setPlayerDetList(null);
            }
            return;
        }

        // Otherwise, reuse them
        FrameRes res = resultsMap.get(frameIdx);
        if (res != null) {
            res.setPlayerDetList(lastPlayerBboxes);
            res.setFrameKps(lastKeypoints);
        }
        retrieveFrameResult();

    }

    private void courtSkipDetection(long frameIdx) {
        if (lastCourtKeypoints == null) {
            FrameRes frameRes = resultsMap.get(frameIdx);
            if (frameRes != null) {
                frameRes.setCourtKps(null);
            }
            return;
        }

        FrameRes res = resultsMap.get(frameIdx);
        if (res != null) {
            res.setCourtKps(lastCourtKeypoints);
        }
        retrieveFrameResult();

    }

    private void runCourtDet(int inputHeight, int inputWidth) throws InterruptedException {
        PreprocessData<Pair<Bitmap, ByteBuffer>> processedData = courtDetInputPq.poll(); // Retrieve the next preprocessed frame
        if (processedData != null) {
            long idx = processedData.getFrameIndex();
            ByteBuffer inputBuffer = processedData.getData().second;

            // Run court detection inference on one thread
            courtExecutor.submit(() -> {
                TensorBuffer outputBuffer = courtDetector.inference(inputBuffer);
                float[] inf = outputBuffer.getFloatArray();
                List<float[]> kps = courtDetector.postprocess(inf, inputHeight, inputWidth);
                Log.d(TAG, "courtKeypoints=" + kps);
                // Store the results for this frame
                FrameRes frameRes = resultsMap.get(idx);
                if (frameRes != null) {
                    frameRes.setCourtKps(kps);
                    Log.d(TAG, "setting courtKeypoints to frameRes=" + kps);
                }

                // Attempt to retrieve completed results
                retrieveFrameResult();
            });
        }
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

            // Get the result
            List<BallPos> ballPositions = res.getBallPositions();
            List<Bbox> bboxes = res.getPlayerDetList();
            List<List<KeyPoint>> frameKps = res.getFrameKps();
            List<float[]> courtKps = res.getCourtKps();

            // Update UI
            if (listener != null) {
                listener.onBallPosCallback(ballPositions);
                listener.onPlayerDetCallback(bboxes);
                listener.onPlayerPoseCallback(frameKps);
                listener.onCourtDetCallback(courtKps);

                // Excute action prediction
                if (frameKps != null && !frameKps.isEmpty()) {
                    // if only track the first player
                   float[] actionProb = playerPoseEstimator.classifyKeypoints(
                            frameKps.get(0),
                            cameraCapturedWidth,
                            cameraCapturedHeight
                    );
                    if (actionProb != null) {
                    // Since the interface does not have an onActionPredictCallback method,
                    // we can handle the action recognition result in other ways.
                    // For example: log the result, or pass it through another callback method.
                        Log.d("FrameAnalyzer", "Action probabilities: S=" + actionProb[0] +
                                ", B=" + actionProb[1] + ", N=" + actionProb[2] + ", F=" + actionProb[3]);
                        listener.onActionPredictCallback(actionProb);
                    }
                }
            }

            // Calculate FPS
            completedFramesCnt.incrementAndGet();
            long currentTic = System.currentTimeMillis();
            if (currentTic - lastTic >= 1000) {
                completedFpsCnt = completedFramesCnt.get();
                completedFramesCnt.set(0);
                lastTic = currentTic;
                Log.d("FrameAnalyzer", "Completed FPS: " + completedFpsCnt);
                if (listener != null) {
                    listener.onPerformanceCallback(completedFpsCnt);
                }
            }
        }
    }


    /**
     * Shuts down all threads and executors used by the pipeline.
     */
    public void shutdownExecutor() {
        PreprocessThreadPool.getInstance().shutdown();
        playerExecutor.shutdownNow();
        tennisExecutor.shutdownNow();
        poseExecutor.shutdown();
        courtExecutor.shutdown();
        inferenceThread.quitSafely();
    }
}