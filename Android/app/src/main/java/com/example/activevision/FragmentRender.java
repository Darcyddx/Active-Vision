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
import com.example.activevision.data.KeyPoint;

import org.opencv.core.Point;

import java.util.ArrayList;
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
    private final Paint mBallPosPaint = new Paint();
    private final Paint mTextColor = new Paint();
    private final Paint mPlayerDetPaint = new Paint();

    private final Paint mKpsPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final Paint mKpsLinePaint = new Paint();

    private final List<List<KeyPoint>> playerKeyPoints = new ArrayList<>();

    private final int[][] skeleton = {
            {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
            {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {0, 1}, {0, 2}, {1, 3}, {2, 4}
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

        mKpsPaint.setColor(Color.GREEN);
        mKpsPaint.setStyle(Paint.Style.FILL);
        mKpsPaint.setStrokeWidth(20 * getWidth() / 800.0f);

        mKpsLinePaint.setAlpha(200);
        // mKpsLinePaint.setStyle(Paint.Style.STROKE);
        mKpsLinePaint.setStyle(Paint.Style.FILL);
        mKpsLinePaint.setAntiAlias(true);
        mKpsLinePaint.setDither(true);
        mKpsLinePaint.setColor(Color.GREEN);
        mKpsLinePaint.setStrokeJoin(Paint.Join.ROUND);
        mKpsLinePaint.setStrokeCap(Paint.Cap.ROUND);
        mKpsLinePaint.setStrokeWidth(3);
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

        float radius = 3f; // Or any radius you prefer
        for (List<KeyPoint> playerKps : playerKeyPoints) {
            for (int i = 0; i < playerKps.size(); i++) {
                float x = (float) playerKps.get(i).getPoint().x;
                float y = (float) playerKps.get(i).getPoint().y;
                // canvas.drawPoint(x, y, mKpsPaint);
                canvas.drawCircle(x, y, radius, mKpsPaint);
            }

            for (int j = 0; j < skeleton.length; j++) {
                float startX = (float) playerKps.get(skeleton[j][0]).getPoint().x;
                float startY = (float) playerKps.get(skeleton[j][0]).getPoint().y;
                float endX = (float) playerKps.get(skeleton[j][1]).getPoint().x;
                float endY = (float) playerKps.get(skeleton[j][1]).getPoint().y;
                if (startX > 0 && startY > 0 && endX > 0 && endY > 0)
                    canvas.drawLine(startX, startY, endX, endY, mKpsLinePaint);
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

    public void renderPlayerKps(@Nullable List<List<KeyPoint>> frameKps, int inputWidth, int inputHeight) {
        // Clear old keypoints.
        playerKeyPoints.clear();

        // If there's nothing to render, just update the view (which clears old content).
        if (frameKps == null) {
            invalidate();
            return;
        }

        float scaleFactor = Math.max((float) getWidth() / inputWidth, (float) getHeight() / inputHeight);

        // Scale each keypoint
        for (List<KeyPoint> onePlayer : frameKps) {
            List<KeyPoint> scaledKps = new ArrayList<>();
            for (KeyPoint kp : onePlayer) {
                float originalX = (float) kp.getPoint().x;
                float originalY = (float) kp.getPoint().y;

                // Scale
                float scaledX = originalX * scaleFactor;
                float scaledY = originalY * scaleFactor;

                // Create a new KeyPoint with scaled coords (or update the existing one).
                // This assumes you have a constructor or setter for (x, y).
                KeyPoint scaledKp = new KeyPoint(kp.getScore(), new Point(scaledX, scaledY));
                scaledKps.add(scaledKp);
            }
            playerKeyPoints.add(scaledKps);
        }

        // Request a redraw
        invalidate();
    }

    public void renderPerformanceInfo(long fps) {
        this.fps = fps;
        invalidate();
    }
}
