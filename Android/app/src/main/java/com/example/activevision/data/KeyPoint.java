package com.example.activevision.data;
import org.opencv.core.Point;

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
