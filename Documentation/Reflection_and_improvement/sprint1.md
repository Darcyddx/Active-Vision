## Reflection on quantizing tennis tracking model TrackNet v2

### The good parts
- The model TrackNet v2 is reduced in size after quantization and significantly decreases memory footprint of deep learning models, making it deployable on mobile devices.
- Execute faster on CPUs with INT8/FP32 and some specialized hardware (TPU/NPU), realizing the feature of real-time.

### The challenges
- Android studio does not provide those machine learning libraries such as numpy. Hence, the calculation of matrices need to be calculated manually.
- The performance is quite poor where it can detect every moving object, it is hard to balance the accuracy and light weight trade-off. 

### Actionable improvement
- Try using TrackNetv4 to see whether the performance is better while maintaining its light weight feature. 

## Reflection on Trial of using TrackNet v4 to replace with the current ball tracking model
### The good parts
- The inference speed of TrackNet v4 is faster than TrackNet v2.

### Challenges
- In some scenarios, the model does not perform very well. There is not a significant improvement compared with TrackNet v2. There is an example showing in the [test video](https://github.com/Darcyddx/Active-Vision/blob/main/Documentation/Reflection_and_improvement/Asset/test_video1.mp4).

### Actionable improvement
- We use the distilled model from ResNet_TrackNet v2 and get better performances. There is an example showing in the [test video](https://github.com/Darcyddx/Active-Vision/blob/main/Documentation/Reflection_and_improvement/Asset/test_video2.mp4).


