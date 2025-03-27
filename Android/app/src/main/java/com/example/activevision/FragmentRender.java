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
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;
import java.util.jar.Attributes;

public class FragmentRender extends View {
    private final ReentrantLock mLock = new ReentrantLock();
    private long fps;
    private List<BallPos> ballPositions = new ArrayList<>();
    private List<Bbox> playerDetBoxes = new ArrayList<>();
    private final Paint mBallPosPaint = new Paint();
    private final Paint mTextColor = new Paint();
    private final Paint mPlayerDetPaint = new Paint();

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

    public void renderPerformanceInfo(long fps) {
        this.fps = fps;
        invalidate();
    }
}
