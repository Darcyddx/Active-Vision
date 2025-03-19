package com.example.active_vision_qualcomm.utils;

import android.graphics.Bitmap;
import android.graphics.Canvas;

import android.graphics.RectF;
import android.util.Pair;

import com.example.active_vision_qualcomm.data.Bbox;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;


public class ImageOps {
    private ImageOps() {}
    /**
     * Adopt from https://github.com/pytorch/android-demo-app/issues/259
     * Image augmentations: first scale the image and later padding image,  increase the strength of the model.
     * Always this scale and padding will make the image object detection gave more high probability or more robust.
     * <p>
     * Reference:
     * https://github.com/ultralytics/yolov5/blob/db6ec66a602a0b64a7db1711acd064eda5daf2b3/utils/augmentations.py#L91-L122
     * def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
     * method
     *
     * @param srcBitmap
     * @param newShape  (640*640)
     * @param auto      default:false, no use
     * @param scaleFill default:false,  no use
     * @param scaleUp   default:false
     * @param stride    default:32 , no use
     * @return
     */
    public static Bitmap letterbox(Bitmap srcBitmap, Pair<Integer, Integer> newShape, Boolean auto,
                                   Boolean scaleFill, Boolean scaleUp, int stride) {
        // current shape
        int currentWidth = srcBitmap.getWidth();
        int currentHeight = srcBitmap.getHeight();

        // new shape eg: 640*640
        int newWidth = newShape.first;
        int newHeight = newShape.second;

        // only scale image，no padding,just return scale image
        // I modify this logic something difference with the python code clean & speed.
        if (scaleFill) {
            // filter =  bilinear filtering
            return Bitmap.createScaledBitmap(srcBitmap, newWidth, newHeight, true);
        }

        // Scale ratio (new / old)
        float r = Math.min(newWidth * 1.0f / currentWidth, newHeight * 1.0f / currentHeight);

        //  Only scale down, do not scale up (for better val mAP)
        if (!scaleUp) {
            r = Math.min(r, 1.0f);
        }

        int newUnpadWidth = Math.round(currentWidth * r);
        int newUnpadHeight = Math.round(currentHeight * r);

        //  wh padding
        int dw = newWidth - newUnpadWidth;
        int dh = newHeight - newUnpadHeight;

        // auto always false, no use for android demo
        if (auto) { // # wh padding
            dw = dw % stride;
            dh = dh % stride;
        }

        // resize
        if (!(currentWidth == newUnpadWidth && currentHeight == newUnpadHeight)) {
            srcBitmap = Bitmap.createScaledBitmap(srcBitmap, newUnpadWidth, newUnpadHeight, true);
        }

        // padding with gray color
        Bitmap outBitmap = Bitmap.createBitmap(srcBitmap.getWidth() + dw, srcBitmap.getHeight() + dh, Bitmap.Config.ARGB_8888);
        Canvas can = new Canvas(outBitmap);
        can.drawRGB(114, 114, 114); // gray color
        // can.drawBitmap(srcBitmap, dw, dh, null);
        int offsetX = dw / 2;
        int offsetY = dh / 2;

        // Draw the resized image in the center
        can.drawBitmap(srcBitmap, offsetX, offsetY, null);

        return outBitmap;
    }

    public static Point getBoxScale(Bbox bbox) {
        final float aspectRatio = 0.75f;
        final float pixelStd = 200.0f;
        final float scaleRatio = 1.25f;

        float width = bbox.getWidth();
        float height = bbox.getHeight();

        if (width > aspectRatio * height) {
            height = width / aspectRatio;
        } else if (width < aspectRatio * height) {
            width = height * aspectRatio;
        }

        float scaleX = (width / pixelStd) * scaleRatio;
        float scaleY = (height / pixelStd) * scaleRatio;
        return new Point(scaleX, scaleY);
    }

    public static Mat getAffineTransMatrix(Point center,
                                           Point scale,
                                           float rot,
                                           Point outputSize,
                                           Point shift,
                                           boolean inv) {
        float scaleTmpX = (float) (scale.x * 200.0);
        double scaleTmpY = (float) (scale.y * 200.0);

        float srcWidth = scaleTmpX;
        float dstWidth = (float) outputSize.x;
        float dstHeight = (float) outputSize.y;

        double rotationRad =  Math.PI * rot / 180.0f;
        double sn = Math.sin(rotationRad);
        double cs = Math.cos(rotationRad);

        float srcPoint0 = 0;
        float srcPoint1 = -0.5f * srcWidth;

        float srcDirX = (float) (srcPoint0 * cs - srcPoint1 * sn);
        float srcDirY = (float) (srcPoint0 * sn + srcPoint1 * cs);
        float dstDirX = 0;
        float dstDirY = -0.5f * dstWidth;

        // Compute the source points.
        Point src0 = new Point(center.x + scaleTmpX * shift.x, center.y + scaleTmpY * shift.y);
        Point src1 = new Point(center.x + srcDirX + scaleTmpX * shift.x, center.y + srcDirY + scaleTmpY * shift.y);
        Point src2 = get3rdPoint(src0, src1);

        // Compute the destination points.
        Point dst0 = new Point(dstWidth * 0.5, dstHeight * 0.5);
        Point dst1 = new Point(dstWidth * 0.5 + dstDirX, dstHeight * 0.5 + dstDirY);
        Point dst2 = get3rdPoint(dst0, dst1);

        // Pack the points into MatOfPoint2f.
        MatOfPoint2f srcPoints = new MatOfPoint2f(src0, src1, src2);
        MatOfPoint2f dstPoints = new MatOfPoint2f(dst0, dst1, dst2);

        // Compute the affine transformation matrix.
        Mat trans;
        if (inv) {
            trans = Imgproc.getAffineTransform(dstPoints, srcPoints);
        } else {
            trans = Imgproc.getAffineTransform(srcPoints, dstPoints);
        }
        return trans;

    }

    public static Point get3rdPoint(Point a, Point b) {
        return new Point(b.x - (a.y - b.y), b.y + (a.x - b.x));
    }

    public static Bbox getActualBox(Bbox box, int maxSide) {
        float left = box.getRect().left * maxSide;
        float top = box.getRect().top * maxSide;
        float right = box.getRect().right * maxSide;
        float bottom = box.getRect().bottom * maxSide;
        float width = right - left;
        float height = bottom - top;
        float cx = left + width * 0.5f;
        float cy = top + height * 0.5f;
        return new Bbox(box.getClsId(), box.getCnf(), cx, cy, width, height,
                new RectF(left, top, right, bottom));
    }

    public static void affineTransform(Point pt, Mat t) {
        // Create a 3x1 Mat (homogeneous coordinates) for the point.
        Mat newPt = new Mat(3, 1, CvType.CV_64FC1);
        newPt.put(0, 0, pt.x);
        newPt.put(1, 0, pt.y);
        newPt.put(2, 0, 1.0);

        // Multiply the affine transform matrix (2x3) by the point (3x1) => (2x1)
        Mat out = new Mat();
        Core.gemm(t, newPt, 1.0, new Mat(), 0.0, out);

        // Extract the transformed (x', y') from the 2x1 result
        double xTrans = out.get(0, 0)[0];
        double yTrans = out.get(1, 0)[0];

        pt.x = xTrans;
        pt.y = yTrans;
    }
}