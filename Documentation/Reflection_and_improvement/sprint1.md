## Reflection on quantizing tennis tracking model TrackNet v2

### The good parts
- The model TrackNet v2 is reduced in size after quantization and significantly decreases memory footprint of deep learning models, making it deployable on mobile devices.
- Execute faster on CPUs with INT8/FP32 and some specialized hardware (TPU/NPU), realizing the feature of real-time.

### The challenges
- Android studio does not provide those machine learning libraries such as numpy. Hence, the calculation of matrices need to be calculated manually.
- The performance is quite poor where it can detect every moving object, it is hard to balance the accuracy and light weight trade-off. 

### Actionable improvement
- Try using TrackNetv4 to see whether the performance is better while maintaining its light weight feature. 

## Reflection on TrackNet v4 test
