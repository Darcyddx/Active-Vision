package com.example.active_vision_qualcomm.data;

import java.nio.ByteBuffer;

public class PreprocessData<T> implements Comparable<PreprocessData<T>> {
    private final long frameIndex;
    private final T data;

    public PreprocessData(long frameIndex, T data) {
        this.frameIndex = frameIndex;
        this.data = data;
    }

    public long getFrameIndex() {
        return frameIndex;
    }

    public T getData() {
        return data;
    }

    @Override
    public int compareTo(PreprocessData<T> other) {
        return Long.compare(this.frameIndex, other.frameIndex);
    }
}
