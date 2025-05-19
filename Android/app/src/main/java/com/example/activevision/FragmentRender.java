package com.example.activevision;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.activevision.data.BallPos;
import com.example.activevision.data.Bbox;
import com.example.activevision.data.KeyPoint;

import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

public class FragmentRender extends View {
    private final ReentrantLock mLock = new ReentrantLock();
    private List<BallPos> ballPositions = new ArrayList<>();

    private List<Bbox> playerDetBoxes = new ArrayList<>();

    private final List<List<KeyPoint>> playerKeyPoints = new ArrayList<>();

    private final float[][] courtKeyPoints = new float[14][2];

    // 添加动作概率相关字段
    private float[] actionProbabilities = null;
    private final Paint mProbBarPaint = new Paint();
    private final String[] actionLabels = {"S", "B", "N", "F"};

    private final int[][] skeleton = {
            {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
            {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {0, 1}, {0, 2}, {1, 3}, {2, 4}
    };

    private final int[][] connections = {
            {0, 1}, {0, 10}, {1, 2}, {1, 4},
            {2, 3}, {2, 6}, {3, 13},
            {4, 5}, {4, 7}, {5, 6}, {5, 8}, {6, 9},
            {7, 11}, {7, 8}, {8, 9}, {9, 12}, {10, 11},
            {11, 12}, {12, 13}
    };

    private long fps;
    private final Paint mBallPosPaint = new Paint();

    private final Paint mPlayerDetPaint = new Paint();

    private final Paint mTextColor = new Paint();

    private final Paint mKpsPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final Paint mKpsLinePaint = new Paint();

    private final Paint courtkpPaint = new Paint();

    private final Paint courtLinePaint = new Paint();
    private Float lastSpeedKmh = null;  // ball speed
    private String lastShotType = null; // ball hit type
    private Paint mActionProbPaint;
    private static final String[] ACTION_LABELS = {"Serve", "Backhand", "Neutral", "Forehand"};
    private static final int[] ACTION_COLORS = {Color.RED, Color.BLUE, Color.GRAY, Color.GREEN};
    private static final String TAG = FragmentRender.class.getSimpleName();// 日志标签

    public FragmentRender(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);

        mBallPosPaint.setColor(Color.RED);
        mBallPosPaint.setStyle(Paint.Style.FILL);
        mBallPosPaint.setAntiAlias(true);
        mBallPosPaint.setStrokeWidth(5f);

        mPlayerDetPaint.setColor(Color.YELLOW); // Set bounding box color
        mPlayerDetPaint.setStyle(Paint.Style.STROKE); // Outline only
        mPlayerDetPaint.setStrokeWidth(3f); // Thickness of the bounding box

        mTextColor.setColor(Color.WHITE);
        mTextColor.setTypeface(Typeface.DEFAULT_BOLD);
        mTextColor.setStyle(Paint.Style.FILL);
        mTextColor.setTextSize(50);

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

        // 初始化概率柱状图画笔
        mProbBarPaint.setColor(Color.RED);
        mProbBarPaint.setStyle(Paint.Style.FILL);
        mProbBarPaint.setAntiAlias(true);
        mProbBarPaint.setAlpha(180); // 设置半透明

        mActionProbPaint = new Paint();
        mActionProbPaint.setTextSize(30);
        mActionProbPaint.setColor(Color.WHITE);

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

        // Draw the bar chart of stroke action probabilities
        if (actionProbabilities != null) {
            int viewW = getWidth();
            // Define the bar chart area size
            float barWidth = 30f;
            float maxBarHeight = 150f; // Maximum bar height in pixels, representing probability 1.0
            float spacing = 15f;
            // Starting position for drawing (slightly offset from the top-right corner)
            float startX = viewW - (barWidth * actionProbabilities.length + spacing * (actionProbabilities.length - 1)) - 20;
            float baseY = 20f; // Y coordinate of the top baseline (20px from the top of the view)

            for (int i = 0; i < actionProbabilities.length; i++) {
                float prob = actionProbabilities[i];
                // Calculate the height of the bar
                float barHeight = prob * maxBarHeight;
                // Construct the coordinates of the bar rectangle
                float left = startX + i * (barWidth + spacing);
                float top = baseY;
                float right = left + barWidth;
                float bottom = top + barHeight;
                // Draw a red rectangle bar
                canvas.drawRect(left, top, right, bottom, mProbBarPaint);
                // Draw the category label
                canvas.drawText(actionLabels[i], left + barWidth / 2 - 10, baseY + maxBarHeight + 40, mTextColor);
            }
        }

        // Draw the bar chart of stroke action probabilities
        int barWidth = 100;
        int barHeight = 20;
        int startX = 50;
        int startY = 50;
        int spacing = 10;

        Log.d(TAG, "ballPositions=" + ballPositions);
        Log.d(TAG, "playerKeyPoints=" + playerKeyPoints);
        Log.d(TAG, "skeleton=" + Arrays.toString(skeleton));       // skeleton 一维打印
        Log.d(TAG, "actionProbabilities=" + Arrays.toString(actionProbabilities));

        if (actionProbabilities != null && ACTION_LABELS != null && ACTION_COLORS != null){
            for (int i = 0; i < 4; i++) {
                // 绘制柱状图
                mActionProbPaint.setColor(ACTION_COLORS[i]);
                canvas.drawRect(startX, startY + i * (barHeight + spacing),
                        startX + (int)(barWidth * actionProbabilities[i]),
                        startY + barHeight + i * (barHeight + spacing), mActionProbPaint);

                // 绘制标签
                mActionProbPaint.setColor(Color.WHITE);
                canvas.drawText(ACTION_LABELS[i] + ": " + String.format("%.2f", actionProbabilities[i]),
                        startX + barWidth + spacing,
                        startY + barHeight/2 + i * (barHeight + spacing), mActionProbPaint);
            }
        }

//        // draw court keypoints
//        for (int i = 0; i < 14; i++) {
//            float x = this.courtKeyPoints[i][0];
//            float y = this.courtKeyPoints[i][1];
//
//            canvas.drawCircle(x, y, 6f, this.courtkpPaint);
//
//        }
//
//        // draw court lines
//        for (int[] line : this.connections) {
//            int i1 = line[0];
//            int i2 = line[1];
//            float x1 = this.courtKeyPoints[i1][0];
//            float y1 = this.courtKeyPoints[i1][1];
//            float x2 = this.courtKeyPoints[i2][0];
//            float y2 = this.courtKeyPoints[i2][1];
//
//            if (x1 >= 0 && x2 >= 0 && y1 >= 0 && y2 >= 0) {
//                canvas.drawLine(x1, y1, x2, y2, this.courtLinePaint);
//            }
//        }
        if (lastSpeedKmh != null && lastShotType != null) {
            String speedInfo = String.format("Speed: %.1f km/h (%s)", lastSpeedKmh, lastShotType);
            canvas.drawText(speedInfo, 50, 120, mTextColor);
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

        // Request redraw of the view
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

//    public void renderCourtPos(float[][][] courtKps, int inputWidth, int inputHeight) {
//        if (courtKps == null) {
//            invalidate();
//            return;
//        }
//
//        // Clear old keypoints
//        for (int i = 0; i < this.courtKeyPoints.length; i++) {
//            Arrays.fill(this.courtKeyPoints[i], 0f);
//        }
//
//        float scaleX = (float) inputWidth / 640f;
//        float scaleY = (float) inputHeight / 640f;
//
//        float[][] keypoints = courtKps[0]; // [14][3]
//
//        for (int i = 0; i < 14; i++) {
//
//            this.courtKeyPoints[i][0] = keypoints[i][0] * scaleX;
//            this.courtKeyPoints[i][1] = keypoints[i][1] * scaleY;
//        }
//
//        // Request redraw of the view
//        invalidate();
//    }

    public void renderPerformanceInfo(long fps) {
        this.fps = fps;
    }

    /**
     * 更新动作概率数据并重绘界面
     * @param probs 长度为4的概率数组[serve, backhand, neutral, forehand]，如果为null则清除显示
     */
    public void renderActionProb(@Nullable float[] probs) {
        mLock.lock();
        if (probs == null) {
            this.actionProbabilities = null;
        } else {
            this.actionProbabilities = probs.clone();
        }
        mLock.unlock();
        invalidate();
    }

    //    public void setActionProbabilities(float[] probabilities) {
//        if (probabilities != null && probabilities.length == 4) {
//            System.arraycopy(probabilities, 0, actionProbabilities, 0, 4);
//            invalidate();
//        }
//    }
    public void setActionProbabilities(float[] probs) {
        if (probs == null) return;
        // 先分配
        this.actionProbabilities = new float[probs.length];
        // 再拷贝
        System.arraycopy(probs, 0, this.actionProbabilities, 0, probs.length);
        invalidate();  // 通知重绘
    }

    public void renderBallSpeed(float speedKmh, String shotType) {
        mLock.lock();
        try {
            this.lastSpeedKmh = speedKmh;
            this.lastShotType = shotType;
        } finally {
            mLock.unlock();
        }
        invalidate();
    }
}
