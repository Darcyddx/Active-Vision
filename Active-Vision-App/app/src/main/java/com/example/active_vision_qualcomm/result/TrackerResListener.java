package com.example.active_vision_qualcomm.result;

import com.example.active_vision_qualcomm.data.BallPos;
import com.example.active_vision_qualcomm.data.Bbox;
import com.example.active_vision_qualcomm.data.KeyPoint;

import java.util.List;

public interface TrackerResListener {
    // to retrieve tennis ball positions
    void onBallPosCallback(List<BallPos> ballPositions);
    // to retrieve player's bbox information
    void onPlayerDetCallback(List<Bbox> bboxes);

    void onPlayerPoseCallback(List<List<KeyPoint>> frameKps);
    // to get fps result
    void onPerformanceCallback(long fps);
    // to get action prediction results
    void onActionPredictCallback(float[] actionProbabilities);
}
