package com.example.activevision.result;

import com.example.activevision.data.BallPos;
import com.example.activevision.data.Bbox;
import com.example.activevision.data.KeyPoint;

import java.util.List;

public interface TrackerResListener {
    // to retrieve tennis ball positions
    void onBallPosCallback(List<BallPos> ballPositions);
    // to get player's location
    void onPlayerDetCallback(List<Bbox> bboxes);
    // to get fps result
    void onPerformanceCallback(long fps);

    void onPlayerPoseCallback(List<List<KeyPoint>> frameKps);

    void onCourtDetCallback(float[][][] courtKps);

    // to get action prediction results
    void onActionPredictCallback(float[] actionProbabilities);
}
