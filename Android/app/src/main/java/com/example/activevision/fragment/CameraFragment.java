package com.example.activevision.fragment;

import android.os.Bundle;
import android.util.Log;
import android.view.Display;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import com.example.activevision.FragmentRender;
import com.example.activevision.FrameAnalyzer;
import com.example.activevision.R;
import com.example.activevision.data.BallPos;
import com.example.activevision.data.Bbox;
import com.example.activevision.data.KeyPoint;
import com.example.activevision.result.TrackerResListener;
import com.example.activevision.trackers.BallTracker;
import com.example.activevision.trackers.PlayerDetector;
import com.example.activevision.trackers.PlayerPoseEstimator;
import com.example.activevision.trackers.PlayerPoseTracker;
import com.example.activevision.trackers.CourtDetector;
import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;

import java.util.List;
import java.util.concurrent.ExecutionException;




/**
 * This class Manage camera interface and life cycle.
 * This is where to process camera video streams and pass
 * frames to analysis or detection algorithms
 * Author: Xi Ding
 * Date: 25/03/2025
 */




public class CameraFragment extends Fragment implements TrackerResListener {

    // UI component to display the camera preview
    private PreviewView mPreviewView;
    // Custom view for rendering detected objects
    private FragmentRender mFragmentRender;

    // TFLite models for tracking
    private BallTracker tennisTracker;

    private PlayerDetector playerDetector;

    private PlayerPoseTracker playerPoseTracker;
    private PlayerPoseEstimator playerPoseEstimator;

    private CourtDetector courtDetector;

    // Analyzer responsible for processing camera frames
    private FrameAnalyzer analyzer;


    public CameraFragment() {
        // Required empty public constructor
    }

    /**
     * Factory method to create a new instance of CameraFragment with specified trackers.
     *
     * @param tracker         The TennisTracker model for ball tracking.
     * @param playerDetector  The PlayerDetector model for player detection.
     * @return A new instance of fragment CameraFragment.
     */
    public static CameraFragment newInstance(BallTracker tracker,
                                             PlayerDetector playerDetector,
                                             PlayerPoseTracker playerPoseTracker,
                                             PlayerPoseEstimator playerPoseEstimator,
                                             CourtDetector courtDetector) {
        CameraFragment fragment = new CameraFragment();
        fragment.tennisTracker = tracker;
        fragment.playerDetector = playerDetector;
        fragment.playerPoseTracker = playerPoseTracker;
        fragment.playerPoseEstimator = playerPoseEstimator;
        fragment.courtDetector = courtDetector;
        return fragment;
    }

    /**
     * Initializes the FrameAnalyzer with the provided trackers and listener.
     * @param savedInstanceState If the fragment is being re-created from
     * a previous saved state, this is the state.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // analyzer = new FrameAnalyzer(tennisTracker, playerDetector, this);
        analyzer = new FrameAnalyzer(tennisTracker, playerDetector, playerPoseTracker,playerPoseEstimator, courtDetector, this);

    }

    private ActivityResultLauncher<String> requestPermissionLauncher;

    /**
     * Called to have the Fragment instantiate its user interface view.
     * Also sets up the permission launcher for camera
     * @param inflater The LayoutInflater object that can be used to inflate
     * any views in the fragment,
     * @param container If non-null, this is the parent view that the fragment's
     * UI should be attached to.  The fragment should not add the view itself,
     * but this can be used to generate the LayoutParams of the view.
     * @param savedInstanceState If non-null, this fragment is being re-constructed
     * from a previous saved state as given here.
     *
     * @return
     */
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        OpenCVLoader.initDebug();

        requestPermissionLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(),
                isGranted -> {
                    if (!isGranted) {
                        // Permission has been denied, show an error dialog or a message
                        Toast.makeText(getContext(),
                                "Camera permission denied",
                                Toast.LENGTH_LONG).show();
                    }
                }
        );
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_camera, container, false);
    }

    /**
     * Initializes UI components for the camera preview and rendering.
     * @param view The View returned by {@link #onCreateView(LayoutInflater, ViewGroup, Bundle)}.
     * @param savedInstanceState If non-null, this fragment is being re-constructed
     * from a previous saved state as given here.
     */
    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        mPreviewView = view.findViewById(R.id.camera_preview);
        mFragmentRender = view.findViewById(R.id.fragmentRender);
    }

    @Override
    public void onResume() {
        super.onResume();

        startCamera(analyzer);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        analyzer.shutdownExecutor();
    }

    /**
     * Sets up and starts the camera, binds it to the lifecycle, and configures the ImageAnalysis.
     * @param analyzer The ImageAnalysis.Analyzer responsible for processing camera frames with ml models
     */
    private void startCamera(ImageAnalysis.Analyzer analyzer) {
        // An instance to bind the lifecycle of cameras to lifecycle owners
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(requireContext());

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // Get the current display rotation for proper image orientation
                Display display = mPreviewView.getDisplay();
                int rotation = (display != null) ? display.getRotation() : Surface.ROTATION_0;

                // Configure ImageAnalysis. Note: this is the frame captured by camera
                // ref: https://developer.android.com/media/camera/camerax/analyze
                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
//                            .setTargetRotation(rotation)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
                imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(requireContext()), analyzer);

                // Configure Preview to display the camera feed on PreviewView
                // Note: This is the view displayed on phone's screen, which is different from
                // the frame captured by camera
                Preview previewBuilder = new Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
//                            .setTargetRotation(rotation)
                        .build();
                previewBuilder.setSurfaceProvider(mPreviewView.getSurfaceProvider());
                Log.i("builder", mPreviewView.getHeight() + "/" + mPreviewView.getWidth());

                // Select the back-facing camera as the default
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                cameraProvider.bindToLifecycle(this, cameraSelector, previewBuilder, imageAnalysis);

            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }

        }, ContextCompat.getMainExecutor(requireContext()));

    }

    @Override
    public void onBallPosCallback(List<BallPos> ballPositions) {
        requireActivity().runOnUiThread(() -> {
            mFragmentRender.renderBallPos(ballPositions,
                    tennisTracker.getInputWidth(),
                    tennisTracker.getInputHeight());
        });
    }

    @Override
    public void onPlayerDetCallback(List<Bbox> bboxes) {
        requireActivity().runOnUiThread(() -> {
            mFragmentRender.renderPlayerPos(bboxes);
        });
    }

    @Override
    public void onPlayerPoseCallback(List<List<KeyPoint>> frameKps) {
        requireActivity().runOnUiThread(() -> {
            mFragmentRender.renderPlayerKps(frameKps,
                    analyzer.getCameraCapturedWidth(),
                    analyzer.getCameraCapturedHeight());
        });
    }

    @Override
    public void onCourtDetCallback(List<float[]> courtKps) {
        requireActivity().runOnUiThread(() -> {
            mFragmentRender.renderCourtPos(courtKps,
                    analyzer.getCameraCapturedWidth(),
                    analyzer.getCameraCapturedHeight());
        });
    }

    @Override
    public void onActionPredictCallback(float[] actionProbabilities) {
        requireActivity().runOnUiThread(() -> {
            mFragmentRender.setActionProbabilities(actionProbabilities);
        });
    }

    @Override
    public void onPerformanceCallback(long fps) {
        requireActivity().runOnUiThread(() -> {
            mFragmentRender.renderPerformanceInfo(fps);
        });
    }
}