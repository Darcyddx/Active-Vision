package com.example.activevision.data;

/**
 *
 * Save the results of a hit detection, including speed, direction, etc.
 * Author: Xingchen Zhang
 * Date: 26/04/2025
 */
public class BallHitResult {
    private final float speedKmh;
    private final String shotType;  // Forehand / Backhand / Unknown

    public BallHitResult(float speedKmh, String shotType) {
        this.speedKmh = speedKmh;
        this.shotType = shotType;
    }

    public float getSpeedKmh() {
        return speedKmh;
    }

    public String getShotType() {
        return shotType;
    }

    @Override
    public String toString() {
        return "BallFlightResult{" +
                "speedKmh=" + speedKmh +
                ", shotType='" + shotType + '\'' +
                '}';
    }
}
