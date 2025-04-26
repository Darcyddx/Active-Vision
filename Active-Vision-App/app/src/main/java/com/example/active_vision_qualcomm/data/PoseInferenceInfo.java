package com.example.active_vision_qualcomm.data;

import org.opencv.core.Point;

public class PoseInferenceInfo {
    private float[] heatmap;
    private Point center;

    private Point scale;

    public PoseInferenceInfo(float[] heatmaps, Point center, Point scale) {
        this.heatmap = heatmaps;
        this.center = center;
        this.scale = scale;
    }

    public float[] getHeatmap() {
        return heatmap;
    }

    public Point getCenter() {
        return center;
    }

    public Point getScale() {
        return scale;
    }
}
