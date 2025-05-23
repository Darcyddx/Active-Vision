### Prerequisites

- Android Studio (version 2022.3 or newer)
- Android SDK 29 and above (Targeted SDK 35)
- Java JDK 11
- Git
- Minimum Android device: Android 10, ARM64 processor

### Clone repository

```(shell)
    git clone https://github.com/Darcyddx/Active-Vision.git
```

### Open project in Android Studio

1. Launch Android Studio
2. Select Open an existing project
3. Navigate to the cloned ```Active-Vision``` directory and open only ```Android``` directory

### Install Dependencies

Android Studio will auto-download Gradle dependencies when Sync

### Build and Run
1. Connect an Android device
2. Click Run ▶️ in Android Studio
3. The app will compile and install on the connected device
4. Find a tennis game video online or YouTube, ensure the camera captures the tennis court


### Targeted Devices

This app supports the following hardware:

- [Qualcomm Hexagon NPU](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
- [GPU -- via GPUv2](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu)
- [CPU -- via XNNPack](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md)

For current implementation, the App expects to run on Qualcomm Snapdragon with Hexagon NPU and should above Snapdragon 8+ Gen 1.

### Tested Devices

| Device Name   | Processor           | Model Delegates | Performance |
| ------------- | ------------------- | --------------- | ----------- |
| Redmi Turbo 3 | Snapdragon 8s Gen 3 | NPU + GPU       | 30 FPS      |
| iQOO Z9 Turbo | Snapdragon 8s Gen 3 | NPU + GPU       | 30 FPS      |
| Xiaomi 15 Pro | Snapdragon 8 Elite  | NPU + GPU       | 30 FPS      |
|               |                     |                 |             |
