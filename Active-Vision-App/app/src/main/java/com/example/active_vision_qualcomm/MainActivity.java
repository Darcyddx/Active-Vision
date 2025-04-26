package com.example.active_vision_qualcomm;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.ProgressBar;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.FragmentTransaction;

import com.example.active_vision_qualcomm.Trackers.PlayerDetector;
import com.example.active_vision_qualcomm.Trackers.PlayerPoseEstimator;
import com.example.active_vision_qualcomm.Trackers.PlayerPoseTracker;
import com.example.active_vision_qualcomm.Trackers.TennisTracker;
import com.example.active_vision_qualcomm.fragment.CameraFragment;
import com.example.active_vision_qualcomm.tflite_helpers.AIHubDefaults;

import java.io.IOException;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {
    private ProgressBar progressBar;

    private TennisTracker tennisTracker;

    private PlayerDetector playerDetector;

    private PlayerPoseTracker playerPoseTracker;
    private PlayerPoseEstimator playerPoseEstimator;

    ExecutorService backgroundTaskExecutor = Executors.newSingleThreadExecutor();
    Handler mainLooperHandler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        progressBar = (ProgressBar)findViewById(R.id.indeterminateBar);

//        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
//            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
//            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
//            return insets;
//        });
        // Load model
        createTFLiteModelAsync();

    }

    /**
     * Method to request Camera permission
     */
    private void cameraPermission() {
        requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
    }

    /**
     * Method to navigate to CameraFragment
     */
    private void overToCamera() {
        boolean passToFragment = MainActivity.this.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
        if (passToFragment) {
            if (tennisTracker != null && playerDetector != null && 
                playerPoseTracker != null && playerPoseEstimator != null) {
                FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
                transaction.add(R.id.main_content, 
                    CameraFragment.newInstance(tennisTracker, playerDetector, 
                                            playerPoseTracker, playerPoseEstimator));
                transaction.commit();
            }
        } else {
            cameraPermission();
        }
    }

    void setLoadingUI(boolean loading) {
        runOnUiThread(() -> progressBar.setVisibility(loading ? View.VISIBLE : View.INVISIBLE));
    }

    /**
     * Create inference objects.
     * Loading the TF Lite model takes time, so this is done asynchronously to the main UI thread.
     * Disables the inference UI during load and reenables it afterwards.
     */
    void createTFLiteModelAsync() {
        if (tennisTracker != null || playerDetector != null) {
            throw new RuntimeException("model was already created");
        }
        setLoadingUI(true);

        // Exit the UI thread and instantiate the model in the background.
        backgroundTaskExecutor.execute(() -> {
            // Create a BallTracker object with all available compute units (NPU, GPU, CPU)
            String tfLiteBallTrackModelAsset = this.getResources().getString(R.string.TrackNetModelAssetUINT8);
            // Create player detector
            String tfLitePlayerDetModelAsset = this.getResources().getString(R.string.PlayerDetModelAssetUINT8);
            String tfLitePlayerPoseModelAsset = this.getResources().getString(R.string.PoseEstMobileModelAssetFP16);
            // 创建动作识别模型
//            String tfLiteActionModelAsset = this.getResources().getString(R.string.TennisRNNModelAssetUINT8);
            String tfLiteActionModelAsset = this.getResources().getString(R.string.PoseEstRNNUnrolledModelAsset);
            
            try {
                tennisTracker = new TennisTracker(
                        this,
                        tfLiteBallTrackModelAsset,
                        AIHubDefaults.delegatePriorityOrder /* AI Hub Defaults */
                );
                playerDetector = new PlayerDetector(
                        this,
                        tfLitePlayerDetModelAsset,
                        AIHubDefaults.delegatePriorityOrder
                );
                playerPoseTracker = new PlayerPoseTracker(
                        this,
                        tfLitePlayerPoseModelAsset,
                        AIHubDefaults.delegatePriorityOrder
                );
                playerPoseEstimator = new PlayerPoseEstimator(
                        this,
                        tfLiteActionModelAsset,
                        AIHubDefaults.delegatePriorityOrder
                );
            } catch (IOException | NoSuchAlgorithmException e) {
                throw new RuntimeException(e.getMessage());
            }
            setLoadingUI(false);
            mainLooperHandler.post(this::overToCamera);
        });
    }


    @Override
    protected void onResume()
    {
        super.onResume();
        overToCamera();
    }
    @Override
    protected void onStop() {
        super.onStop();
    }


}