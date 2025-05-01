package com.example.activevision.result;

import com.example.activevision.data.BallPos;
import com.example.activevision.data.Bbox;
import com.example.activevision.data.KeyPoint;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class FrameRes {
    public final long frameId;

    volatile List<BallPos> ballPosList;

    volatile List<Bbox> playerDetList;

    volatile List<List<KeyPoint>> frameKps;

    private final AtomicInteger numTasks;

    private final AtomicInteger taskCompleted = new AtomicInteger(0);


    public FrameRes(long frameId) {
        this.frameId = frameId;

        if (frameId % 3 == 0) {
            this.numTasks =  new AtomicInteger(3);
        } else {
            this.numTasks =  new AtomicInteger(2);
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

    public synchronized void setPlayerDetList(List<Bbox> bboxes) {
        this.playerDetList = bboxes;
        taskCompleted.incrementAndGet();
    }

    public synchronized void setFrameKps(List<List<KeyPoint>> frameKps) {
        this.frameKps = frameKps;
        taskCompleted.incrementAndGet();
    }

    public List<Bbox> getPlayerDetList() {
        return playerDetList == null ? List.of() : playerDetList;
    }

    public List<List<KeyPoint>> getFrameKps() {
        return frameKps;
    }

    public boolean isComplete() {
        return taskCompleted.get() == numTasks.get();
    }
}
