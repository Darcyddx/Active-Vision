package com.example.activevision.result;

import com.example.activevision.data.BallPos;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class FrameRes {
    public final long frameId;

    volatile List<BallPos> ballPosList;

    private final AtomicInteger numTasks;

    private final AtomicInteger taskCompleted = new AtomicInteger(0);


    public FrameRes(long frameId) {
        this.frameId = frameId;

        if (frameId % 3 == 0) {
            this.numTasks =  new AtomicInteger(1);
        } else {
            this.numTasks =  new AtomicInteger(0);
        }
    }

    public synchronized void setBallPosList(List<BallPos> positions) {
        this.ballPosList = positions;

        if (frameId % 3 == 0) {
            taskCompleted.incrementAndGet();
        }
    }

    public List<BallPos> getBallPositions() {
        return ballPosList == null ? List.of() : ballPosList;
    }

    public boolean isComplete() {
        return taskCompleted.get() == numTasks.get();
    }
}
