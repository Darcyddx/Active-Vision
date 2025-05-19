package com.example.activevision.data;

import org.opencv.core.Point;

/**
 * Represents a keypoint in an image.
 * A keypoint is a point of interest that is detected in an image, often used in
 * computer vision tasks such as object recognition, image matching, and
 * feature tracking. Each keypoint has a score and a coordinate.
 * Author: Zhiyuan Lu
 * Date: 24/03/2025
 */
public class KeyPoint {
    private float score;
    private Point point;

    public KeyPoint() {}

    public KeyPoint(float score, Point point) {
        this.score = score;
        this.point = point;
    }

    public float getScore() {
        return score;
    }

    public void setScore(float score) {
        this.score = score;
    }

    public Point getPoint() {
        return point;
    }

    public void setPoint(Point point) {
        this.point = point;
    }
}
