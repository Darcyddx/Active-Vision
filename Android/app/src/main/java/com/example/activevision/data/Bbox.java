package com.example.activevision.data;

import android.graphics.RectF;

public class Bbox {
    private int clsId;
    private float cnf;

    private float cx;

    private float cy;
    private float width;

    private float height;
    private RectF rect;

    public Bbox(int clsId, float cnf, float cx, float cy, float width, float height, RectF rect) {
        this.clsId = clsId;
        this.cnf = cnf;
        this.cx = cx;
        this.cy = cy;
        this.width = width;
        this.height = height;
        this.rect = rect;
    }

    public int getClsId() {
        return clsId;
    }

    public float getWidth() {
        return width;
    }

    public float getCx() {
        return cx;
    }

    public float getCy() {
        return cy;
    }


    public float getHeight() {
        return height;
    }

    public float getCnf() {
        return cnf;
    }

    public RectF getRect() {
        return rect;
    }

    @Override
    public String toString() {
        return "Bbox{" +
                "clsId=" + clsId +
                ", cnf=" + cnf +
                ", rect=" + rect +
                '}';
    }
}
