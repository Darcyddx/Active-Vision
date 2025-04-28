package com.example.activevision.result;

import com.example.activevision.data.BallPos;
import com.example.activevision.data.Bbox;

import java.util.List;

public interface TrackerResListener {
    // to retrieve tennis ball positions
    void onBallPosCallback(List<BallPos> ballPositions);
    // to get player's location
    void onPlayerDetCallback(List<Bbox> bboxes);
    // to get fps result
    void onPerformanceCallback(long fps);

    void onShotInfoCallback(String shotType, float speedKmh);
}
