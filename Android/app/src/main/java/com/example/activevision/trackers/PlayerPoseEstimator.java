package com.example.activevision.trackers;

import android.content.Context;
import android.util.Pair;

import com.example.activevision.data.KeyPoint;
import com.example.activevision.tflite_helpers.AIHubDefaults;
import com.example.activevision.tflite_helpers.TFLiteHelpers;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * PlayerPoseEstimator is a class that utilizes a TensorFlow Lite model to estimate
 * and classify the pose of a player based on keypoints detected in a sequence of frames.
 * It maintains a buffer of recent keypoint sequences, processes them, and outputs
 * probabilities for different actions (e.g., serve, backhand, neutral, forehand).
 *
 * The class handles both floating-point and quantized models, managing input and
 * output tensor data types, shapes, scales, and zero points.
 *
 * This is mainly for tennis player, to guess the current gesture.
 * Auther: Yichi Zhang
 * Date: 26/04/2025
 */
public class PlayerPoseEstimator implements AutoCloseable {
    private Interpreter tfLiteInterpreter;
    private DataType inputType;
    private DataType outputType;
    private int[] inputShape;
    private int[] outputShape;
    private float inputScale = 1.0f;
    private int inputZeroPoint = 0;
    private float outputScale = 1.0f;
    private int outputZeroPoint = 0;

    /**
     * Buffer to store the most recent 30 frames of keypoint sequences.
     */
    private final List<float[]> sequenceBuffer = new ArrayList<>();
    /**
     * The result of the last calculation is the probability results.
     * @param context Application context.
     * @param modelPath Path to the TFLite model file.
     * @param delegatePriorityOrder Priority order of TFLite delegates to use.
     * @throws IOException If there is an error reading the model file.
     */
    private float[] lastProbabilities = null;

    public PlayerPoseEstimator(Context context, String modelPath,
                               TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {
        // Load the TFLite model file
        // Assume the model file is stored in assets, and use the TFLiteHelpers utility to load it
         Pair<MappedByteBuffer, String> modelAndHash = TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);

        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                modelAndHash.first,
                delegatePriorityOrder,
                AIHubDefaults.numCPUThreads,
                context.getApplicationInfo().nativeLibraryDir,
                context.getCacheDir().getAbsolutePath(),
                modelAndHash.second
        );
        tfLiteInterpreter = iResult.first;
        // Get model input/output tensor information
        Tensor inTensor = tfLiteInterpreter.getInputTensor(0);
        Tensor outTensor = tfLiteInterpreter.getOutputTensor(0);
        inputType = inTensor.dataType();
        outputType = outTensor.dataType();
        inputShape = inTensor.shape();    // should be [1, 30, 26]
        outputShape = outTensor.shape();  // should be [1, 4]
        // If the model is quantized, record the quantization parameters (scale and zeroPoint)
        if (inputType == DataType.UINT8 || inputType == DataType.INT8) {
            inputScale = inTensor.quantizationParams().getScale();
            inputZeroPoint = inTensor.quantizationParams().getZeroPoint();
        }
        if (outputType == DataType.UINT8 || outputType == DataType.INT8) {
            outputScale = outTensor.quantizationParams().getScale();
            outputZeroPoint = outTensor.quantizationParams().getZeroPoint();
        }
    }

    /**
     * Adds a frame's keypoints and performs action classification when the buffer is full (30 frames).
     *
     * @param keypoints   List of keypoints detected in the current frame (should contain 17 keypoints, including nose,
     *                    shoulders, elbows, wrists, hips, knees, ankles, etc.).
     * @param frameWidth  Width of the current frame image (used for normalizing coordinates).
     * @param frameHeight Height of the current frame image.
     * @return A 4-element probability array [serve, backhand, neutral, forehand] if inference is performed;
     *         otherwise, returns null.
     */
    public synchronized float[] classifyKeypoints(List<KeyPoint> keypoints, int frameWidth, int frameHeight) {
        if (keypoints == null || keypoints.isEmpty()) {
            // 若当前帧无人体关键点，清空缓冲避免旧数据干扰
            sequenceBuffer.clear();
            lastProbabilities = null;
            return null;
        }

        // 提取13个关键点（去除眼睛耳朵等），构建长度26的特征向量
        int[] usefulIndices = {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; // 鼻子和主要关节点
        float[] featureVec = new float[usefulIndices.length * 2];
        int j = 0;
        for (int idx : usefulIndices) {
            KeyPoint kp = keypoints.get(idx);
            // 获取坐标
            float x = (float) kp.getPoint().x;
            float y = (float) kp.getPoint().y;
            // 如果需要，进行归一化，将坐标缩放到0~1区间
            x /= frameWidth;
            y /= frameHeight;
            featureVec[j++] = x;
            featureVec[j++] = y;
        }

        // 将该帧特征加入序列缓冲
        sequenceBuffer.add(featureVec);
        if (sequenceBuffer.size() > 30) {
            // 若超过30帧，移除最早的一帧，实现滑动窗口
            sequenceBuffer.remove(0);
        }

        // 未达到30帧，不进行推理
        if (sequenceBuffer.size() < 30) {
            return null;
        }

        // 构建输入张量 [1,30,26]
        // 将30帧特征复制到一个一维数组或直接创建多维数组
        float[][][] inputSeq = new float[1][30][26];
        for (int t = 0; t < 30; t++) {
            inputSeq[0][t] = sequenceBuffer.get(t);
        }
        // 执行推理，根据模型输入类型选择合适的数据格式
        if (inputType == DataType.FLOAT32) {
            // 浮点模型，直接输入浮点数组
            float[][] output = new float[1][4];
            tfLiteInterpreter.run(inputSeq, output);
            lastProbabilities = output[0];
        } else {
            // 量化模型，需要将浮点输入转换为uint8/int8
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(30 * 26);  // 780字节
            inputBuffer.order(ByteOrder.nativeOrder());
            for (int t = 0; t < 30; t++) {
                for (int k = 0; k < 26; k++) {
                    // 将浮点按scale和zeroPoint量化为uint8
                    int quantVal = Math.round(sequenceBuffer.get(t)[k] / inputScale + inputZeroPoint);
                    // 确保在0~255范围内
                    if (quantVal < 0) quantVal = 0;
                    if (quantVal > 255) quantVal = 255;
                    inputBuffer.put((byte) quantVal);
                }
            }
            // 准备输出缓冲区
            ByteBuffer outputBuffer = ByteBuffer.allocateDirect(4);
            outputBuffer.order(ByteOrder.nativeOrder());
            tfLiteInterpreter.run(inputBuffer, outputBuffer);
            
            outputBuffer.rewind();
            // 将输出缓冲区的4个字节解量化为概率值
            lastProbabilities = new float[4];
            for (int i = 0; i < 4; i++) {
                int quantOut = outputBuffer.get() & 0xFF;  // 读出无符号uint8值
                lastProbabilities[i] = (quantOut - outputZeroPoint) * outputScale;
            }
        }
        return lastProbabilities;
    }

    /**
     * Gets the probability results of the most recent inference for the hitting action.
     *
     * @return The most recent array of action probabilities.
     *         Returns null if no inference has been performed yet.
     */
    public synchronized float[] getLastProbabilities() {
        return lastProbabilities;
    }

    @Override
    public void close() {
        if (tfLiteInterpreter != null) {
            tfLiteInterpreter.close();
            tfLiteInterpreter = null;
        }
    }
}
