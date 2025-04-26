package com.example.active_vision_qualcomm.data;

import org.opencv.core.Point;

import java.nio.ByteBuffer;

public class PosePreprocessInfo {
    private ByteBuffer byteBuffer;
    private Point center;
    private Point scale;

    public PosePreprocessInfo(ByteBuffer byteBuffer, Point center, Point scale) {
        this.byteBuffer = byteBuffer;
        this.center = center;
        this.scale = scale;
    }

    public ByteBuffer getByteBuffer() {
        return byteBuffer;
    }

    public Point getCenter() {
        return center;
    }

    public Point getScale() {
        return scale;
    }
}
