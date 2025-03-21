package com.example.activevision.result;

import com.example.activevision.data.BallPos;

import java.util.List;

public interface TrackerResListener {
    // to retrieve tennis ball positions
    void onBallPosCallback(List<BallPos> ballPositions);
    // to get fps result
    void onPerformanceCallback(long fps);
}
