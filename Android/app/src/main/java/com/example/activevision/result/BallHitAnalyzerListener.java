package com.example.activevision.result;

import com.example.activevision.data.BallHitResult;

/**
 * Callback interface for FrameAnalyzer to notify the batting detection result
 */
public interface BallHitAnalyzerListener {
    void onHitDetected(BallHitResult result);
}