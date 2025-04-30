package com.example.activevision;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.activevision.data.BallPos;
import com.example.activevision.data.Bbox;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The class includes methods for setting and updating the positions of tennis balls and player bounding boxes, which are
 * scaled and rendered according to the dimensions of the view. It also displays the frames per second (fps) rate to help
 * monitor the performance of the tracking system in real-time.
 * Author: Zhiyuan Lu
 * Date: 21/03/2025
 */

public class FragmentRender extends View {
    private final ReentrantLock mLock = new ReentrantLock();
    private long fps;
    private List<BallPos> ballPositions = new ArrayList<>();
    private List<Bbox> playerDetBoxes = new ArrayList<>();
    private final float[][] courtKeyPoints = new float[14][2];
    private final Paint mBallPosPaint = new Paint();
    private final Paint mTextColor = new Paint();
    private final Paint mPlayerDetPaint = new Paint();
    private final Paint courtkpPaint = new Paint();
    private final Paint courtLinePaint = new Paint();

    private final int[][] connections = {
            {0, 1}, {0, 10}, {1, 2}, {1, 4},
            {2, 3}, {2, 6}, {3, 13},
            {4, 5}, {4, 7}, {5, 6}, {5, 8}, {6, 9},
            {7, 11}, {7, 8}, {8, 9}, {9, 12}, {10, 11},
            {11, 12}, {12, 13}
    };

    public FragmentRender(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        mBallPosPaint.setColor(Color.RED);
        mBallPosPaint.setStyle(Paint.Style.FILL);
        mBallPosPaint.setAntiAlias(true);
        mBallPosPaint.setStrokeWidth(5f);

        mTextColor.setColor(Color.WHITE);
        mTextColor.setTypeface(Typeface.DEFAULT_BOLD);
        mTextColor.setStyle(Paint.Style.FILL);
        mTextColor.setTextSize(50);

        mPlayerDetPaint.setColor(Color.YELLOW); // Set bounding box color
        mPlayerDetPaint.setStyle(Paint.Style.STROKE); // Outline only
        mPlayerDetPaint.setStrokeWidth(5f); // Thickness of the bounding box

        courtkpPaint.setColor(Color.GREEN);
        courtkpPaint.setStyle(Paint.Style.FILL);

        courtLinePaint.setColor(Color.GREEN);
        courtLinePaint.setStyle(Paint.Style.STROKE);
        courtLinePaint.setStrokeWidth(4f);


    }

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);
        mLock.lock();
        for (BallPos pos : ballPositions) {
            if (pos != null) {
                canvas.drawCircle(pos.getX(), pos.getY(), 10f, mBallPosPaint);
            }
        }
        for (Bbox box: this.playerDetBoxes) {
            if (box != null) {
                canvas.drawRect(box.getRect(), mPlayerDetPaint);
            }
        }

        // draw court keypoints
        for (int i = 0; i < 14; i++) {
            float x = this.courtKeyPoints[i][0];
            float y = this.courtKeyPoints[i][1];

            canvas.drawCircle(x, y, 6f, this.courtkpPaint);

        }

        // draw court lines
        for (int[] line : this.connections) {
            int i1 = line[0];
            int i2 = line[1];
            float x1 = this.courtKeyPoints[i1][0];
            float y1 = this.courtKeyPoints[i1][1];
            float x2 = this.courtKeyPoints[i2][0];
            float y2 = this.courtKeyPoints[i2][1];

            if (x1 >= 0 && x2 >= 0 && y1 >= 0 && y2 >= 0) {
                canvas.drawLine(x1, y1, x2, y2, this.courtLinePaint);
            }
        }


        canvas.drawText("FPS: " + fps, 50, 50, mTextColor);
        mLock.unlock();
    }

    public void renderBallPos(List<BallPos> ballPos, int inputWidth, int inputHeight) {
        if (ballPos == null) {
            invalidate();
            return;
        }
        this.ballPositions.clear();
        // Scale ball positions to match the PreviewView dimensions
        float scaleFactor = Math.max((float) getWidth() / inputWidth, (float) getHeight() / inputHeight);
        for (BallPos pos : ballPos) {
            if (pos != null) {
                int scaledX = Math.round(pos.getX() * scaleFactor);
                int scaledY = Math.round(pos.getY() * scaleFactor);
                this.ballPositions.add(new BallPos(scaledX, scaledY));
            }
        }
        invalidate();
    }

    public void renderPlayerPos(List<Bbox> bboxes) {
        if (bboxes == null) {
            invalidate();
            return;
        }
        this.playerDetBoxes.clear();
        int maxSide = Math.max(getWidth(), getHeight());
        for (Bbox box: bboxes) {
            float left = box.getRect().left * maxSide;
            float top = box.getRect().top * maxSide;
            float right = box.getRect().right * maxSide;
            float bottom = box.getRect().bottom * maxSide;
            float width = right - left;
            float height = bottom - top;
            float cx = left + width * 0.5f;
            float cy = top + height * 0.5f;
            this.playerDetBoxes.add(new Bbox(box.getClsId(), box.getCnf(), cx, cy, width, height,
                    new RectF(left, top, right, bottom)));
        }
        invalidate();
    }

    public void renderCourtPos(float[][][] courtKps, int inputWidth, int inputHeight) {
        if (courtKps == null) {
            invalidate();
            return;
        }

        // Clear old keypoints
        for (int i = 0; i < this.courtKeyPoints.length; i++) {
            Arrays.fill(this.courtKeyPoints[i], 0f);
        }

        float scaleX = (float) inputWidth / 640f;
        float scaleY = (float) inputHeight / 640f;

        float[][] keypoints = courtKps[0]; // [14][3]

        for (int i = 0; i < 14; i++) {

            this.courtKeyPoints[i][0] = keypoints[i][0] * scaleX;
            this.courtKeyPoints[i][1] = keypoints[i][1] * scaleY;
        }

        // Request redraw of the view
        invalidate();
    }

    public void renderPerformanceInfo(long fps) {
        this.fps = fps;
        invalidate();
    }
}
