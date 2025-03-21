package com.example.activevision;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.ProgressBar;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.fragment.app.FragmentTransaction;

import com.example.activevision.fragment.CameraFragment;
import com.example.activevision.tflite_helpers.AIHubDefaults;
import com.example.activevision.trackers.BallTracker;

import java.io.IOException;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private ProgressBar progressBar;

    private BallTracker tennisTracker;

    ExecutorService backgroundTaskExecutor = Executors.newSingleThreadExecutor();
    Handler mainLooperHandler = new Handler(Looper.getMainLooper());


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        progressBar = (ProgressBar)findViewById(R.id.indeterminateBar);
        createTFLiteModelAsync();
//        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
//            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
//            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
//            return insets;
//        });

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
            if (tennisTracker != null) {
                FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
                transaction.add(R.id.main_content, CameraFragment.newInstance(tennisTracker));
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
        if (tennisTracker != null) {
            throw new RuntimeException("model was already created");
        }
        setLoadingUI(true);

        // Exit the UI thread and instantiate the model in the background.
        backgroundTaskExecutor.execute(() -> {
            // Create a BallTracker object with all available compute units (NPU, GPU, CPU)
            String tfLiteBallTrackModelAsset = this.getResources().getString(R.string.TrackNetModelAssetUINT8);
            // Create player detector

            try {
                tennisTracker = new BallTracker(
                        this,
                        tfLiteBallTrackModelAsset,
                        AIHubDefaults.delegatePriorityOrder /* AI Hub Defaults */
                );
//                playerDetector = new PlayerDetector(
//                        this,
//                        tfLitePlayerDetModelAsset,
//                        AIHubDefaults.delegatePriorityOrder
//                );
//                playerPoseTracker = new PlayerPoseTracker(
//                        this,
//                        tfLitePlayerPoseModelAsset,
//                        AIHubDefaults.delegatePriorityOrder
//                );
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