package com.example.activevision.result;

import com.example.activevision.data.BallPos;
import com.example.activevision.data.Bbox;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class FrameRes {
    public final long frameId;

    volatile List<BallPos> ballPosList;

    volatile List<Bbox> playerDetList;

    volatile float[][][] courtKps;

    private final AtomicInteger numTasks;

    private final AtomicInteger taskCompleted = new AtomicInteger(0);


    public FrameRes(long frameId) {
        this.frameId = frameId;

        if (frameId % 3 == 0) {
            this.numTasks =  new AtomicInteger(2);
        } else {
            this.numTasks =  new AtomicInteger(1);
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

    public synchronized void setCourtKps(float[][][] courtKps) {
        this.courtKps = courtKps;
        taskCompleted.incrementAndGet();
    }

    public List<Bbox> getPlayerDetList() {
        return playerDetList == null ? List.of() : playerDetList;
    }

    public float[][][] getCourtKps() {
        return courtKps;
    }

    public boolean isComplete() {
        return taskCompleted.get() == numTasks.get();
    }
}
