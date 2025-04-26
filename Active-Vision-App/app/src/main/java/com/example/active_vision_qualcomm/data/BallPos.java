package com.example.active_vision_qualcomm.data;

public class BallPos {
    private int x;
    private int y;

    public BallPos(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    @Override
    public String toString() {
        return "BallPos{" +
                "x=" + x +
                ", y=" + y +
                '}';
    }
}
