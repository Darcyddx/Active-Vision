package com.example.activevision;

import com.example.activevision.data.BallHitResult;
import com.example.activevision.data.BallPos;
import com.example.activevision.result.BallHitAnalyzerListener;

import java.util.ArrayList;
import java.util.List;

// updated the BallHitAnalyzer code by fixing the timestamp usage to consistently use the passed parameter instead of System.currentTimeMillis(), 
// added a PIXEL_TO_METER_RATIO constant for calibration between pixel and real-world measurements, 
// improved code structure with better comments for clarity, and maintained the original shot type inference logic as requested.
// 5 frames calculation change to 10
// add more choices for speed calculation (default km/h)

public class BallHitAnalyzer {
    /**
     * Available speed units for the calculation
     */
    public enum SpeedUnit {
        KMH,  // Kilometers per hour
        MPH,  // Miles per hour
        MPS   // Meters per second
    }
    
    private static class BallRecord {
        float x;
        float y;
        long timestamp;
        BallRecord(float x, float y, long timestamp) {
            this.x = x;
            this.y = y;
            this.timestamp = timestamp;
        }
    }
    
    private final List<BallRecord> history = new ArrayList<>();
    private static final int MAX_HISTORY_SIZE = 40;
    private static final int SPEED_CALCULATION_FRAMES = 10; // Calculate speed using 10 frames
    private static final long MIN_HIT_INTERVAL_MS = 300;
    private static final float PIXEL_TO_METER_RATIO = 11f;
    private static final float MIN_SPEED_KMH_THRESHOLD = 1.0f;
    private long lastHitTime = 0;
    private BallHitAnalyzerListener listener;
    private SpeedUnit speedUnit = SpeedUnit.KMH; // Default unit is KMH
    
    public BallHitAnalyzer(BallHitAnalyzerListener listener) {
        this.listener = listener;
    }
    
    /**
     * Set the speed unit to be used for calculations
     * @param unit The speed unit (KMH, MPH, or MPS)
     */
    public void setSpeedUnit(SpeedUnit unit) {
        this.speedUnit = unit;
    }
    
    /**
     * Get the current speed unit being used
     * @return The current speed unit
     */
    public SpeedUnit getSpeedUnit() {
        return this.speedUnit;
    }
    
    /**
     * Update the position of the ball in each frame and automatically detect whether the ball is hit.
     * @param ballPos  The ball currently detected
     * @param timestamp Current frame timestamp (milliseconds)
     */
    public void update(BallPos ballPos, long timestamp) {
        if (ballPos == null) {
            return;
        }
        
        BallRecord current = new BallRecord(ballPos.getX(), ballPos.getY(), timestamp);
        history.add(current);
        
        if (history.size() > MAX_HISTORY_SIZE) {
            history.remove(0);
        }
        
        if (detectHit(timestamp)) {
            float speed = calculateSpeed();
            if (speed > 0) {
                String shotType = inferShotType();
                BallHitResult result = new BallHitResult(speed, shotType);
                if (listener != null) {
                    listener.onHitDetected(result);
                }
            }
            lastHitTime = timestamp;
        }
    }
    
    private boolean detectHit(long currentTimestamp) {
        if (history.size() < 5) {
            return false;
        }
        
        BallRecord last = history.get(history.size() - 1);
        BallRecord secondLast = history.get(history.size() - 2);
        float dx1 = secondLast.x - history.get(history.size() - 3).x;
        float dy1 = secondLast.y - history.get(history.size() - 3).y;
        float dx2 = last.x - secondLast.x;
        float dy2 = last.y - secondLast.y;
        
        float dotProduct = dx1 * dx2 + dy1 * dy2;
        if (dotProduct < 0 && (currentTimestamp - lastHitTime) > MIN_HIT_INTERVAL_MS) {
            return true;
        }
        
        return false;
    }
    
    private float calculateSpeed() {
        // Ensure we have enough history for the calculation
        if (history.size() < SPEED_CALCULATION_FRAMES) {
            return -1;
        }
        
        // Use the last 10 frames to calculate speed
        BallRecord first = history.get(history.size() - SPEED_CALCULATION_FRAMES);
        BallRecord last = history.get(history.size() - 1);
        
        float dx = last.x - first.x;
        float dy = last.y - first.y;
        long dt = last.timestamp - first.timestamp;
        
        if (dt <= 0) {
            return -1;
        }
        
        float distancePixels = (float) Math.sqrt(dx * dx + dy * dy);
        float distanceMeters = distancePixels / PIXEL_TO_METER_RATIO;
        float speedMps = distanceMeters / (dt / 1600.0f);
        float speedKmh = speedMps * 3.6f;

        //  printlog
        android.util.Log.d("BallSpeed", "dx=" + dx + ", dy=" + dy + ", dist(px)=" + distancePixels +
                ", dist(m)=" + distanceMeters + ", dt(ms)=" + dt + ", speed(kmh)=" + speedKmh);

        if (speedKmh < MIN_SPEED_KMH_THRESHOLD) {
            return -1;
        }

        switch (speedUnit) {
            case KMH:
                return speedKmh;
            case MPH:
                return speedMps * 2.23694f;
            case MPS:
                return speedMps;
            default:
                return speedKmh;
        }
    }
    
    private String inferShotType() {
        return "Forehand"; // to be modified later
    }
    
    public void reset() {
        history.clear();
        lastHitTime = 0;
    }
}




// public class BallHitAnalyzer{

//     private static class BallRecord {
//         float x;
//         float y;
//         long timestamp;

//         BallRecord(float x, float y, long timestamp) {
//             this.x = x;
//             this.y = y;
//             this.timestamp = timestamp;
//         }
//     }

//     private final List<BallRecord> history = new ArrayList<>();
//     private static final int MAX_HISTORY_SIZE = 30;
//     private static final long MIN_HIT_INTERVAL_MS = 300;
//     private long lastHitTime = 0;

//     private BallHitAnalyzerListener listener;

//     public BallHitAnalyzer(BallHitAnalyzerListener listener) {
//         this.listener = listener;
//     }

//     /**
//      * Update the position of the ball in each frame and automatically detect whether the ball is hit.
//      * @param ballPos  The ball currently detected
//      * @param timestamp Current frame timestamp (milliseconds)
//      */
//     public void update(BallPos ballPos, long timestamp) {
//         if (ballPos == null) {
//             return;
//         }
//         BallRecord current = new BallRecord(ballPos.getX(), ballPos.getY(), timestamp);
//         history.add(current);

//         if (history.size() > MAX_HISTORY_SIZE) {
//             history.remove(0);
//         }

//         if (detectHit()) {
//             float speed = calculateSpeed();
//             if (speed > 0) {
//                 String shotType = inferShotType();
//                 BallHitResult result = new BallHitResult(speed, shotType);
//                 if (listener != null) {
//                     listener.onHitDetected(result);
//                 }
//             }
//             lastHitTime = timestamp;
//         }
//     }

//     private boolean detectHit() {
//         if (history.size() < 5) {
//             return false;
//         }

//         BallRecord last = history.get(history.size() - 1);
//         BallRecord secondLast = history.get(history.size() - 2);

//         float dx1 = secondLast.x - history.get(history.size() - 3).x;
//         float dy1 = secondLast.y - history.get(history.size() - 3).y;
//         float dx2 = last.x - secondLast.x;
//         float dy2 = last.y - secondLast.y;

//         float dotProduct = dx1 * dx2 + dy1 * dy2;
//         if (dotProduct < 0 && (System.currentTimeMillis() - lastHitTime) > MIN_HIT_INTERVAL_MS) {
//             return true;
//         }
//         return false;
//     }

//     private float calculateSpeed() {
//         if (history.size() < 5) {
//             return -1;
//         }

//         BallRecord first = history.get(history.size() - 5);
//         BallRecord last = history.get(history.size() - 1);

//         float dx = last.x - first.x;
//         float dy = last.y - first.y;
//         long dt = last.timestamp - first.timestamp;

//         if (dt <= 0) {
//             return -1;
//         }

//         float distancePixels = (float) Math.sqrt(dx * dx + dy * dy);
//         float distanceMeters = distancePixels / 1000.0f;
//         float speedMps = distanceMeters / (dt / 1000.0f);
//         float speedKmh = speedMps * 3.6f;

//         return speedKmh;
//     }

//     private String inferShotType() {
//         return "Forehand"; // to be modified later
//     }

//     public void reset() {
//         history.clear();
//         lastHitTime = 0;
//     }
// }
