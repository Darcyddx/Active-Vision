package com.example.active_vision_qualcomm.result;

import com.example.active_vision_qualcomm.data.BallPos;
import com.example.active_vision_qualcomm.data.Bbox;
import com.example.active_vision_qualcomm.data.KeyPoint;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class FrameRes {
    public final long frameId;

    volatile List<BallPos> ballPosList;

    volatile List<Bbox> playerDetList;

    volatile List<List<KeyPoint>> frameKps;

    private final AtomicInteger numTasks;

    private final AtomicInteger taskCompleted = new AtomicInteger(0);

    private final long entryTic;

    private static final long FRAME_DROP_THRESHOLD = 1000;

    public FrameRes(long frameId) {
        this.frameId = frameId;
        this.entryTic = System.currentTimeMillis();
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

    public synchronized void setPlayerDetList(List<Bbox> bboxes) {
        this.playerDetList = bboxes;
        taskCompleted.incrementAndGet();
    }

    public synchronized void setFrameKps(List<List<KeyPoint>> frameKps) {
        this.frameKps = frameKps;
        taskCompleted.incrementAndGet();
    }


    public List<BallPos> getBallPositions() {
        return ballPosList == null ? List.of() : ballPosList;
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

    public boolean isExpired() {
        long newTic = System.currentTimeMillis();
        return (newTic - entryTic) >= FRAME_DROP_THRESHOLD;
    }
}
