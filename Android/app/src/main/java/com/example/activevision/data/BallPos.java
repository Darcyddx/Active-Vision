package com.example.activevision.data;


/**
 * This class is designed as an entity class
 * to represent a ball's position
 * Author: Xingchen Zhang
 * Date: 24/03/2025
 */
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
