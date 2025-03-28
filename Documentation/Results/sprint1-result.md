## Complete Tasks

### Player Detection On Android

We have integrated the player detection model (`YOLOv8n`) alongside `TrackNet` on Android devices, achieving real-time performance at 30FPS.  This is achieved by our innovative three-level parallel processing pipeline architecture that optimizes computational resource utilization. The system implements concurrent execution of 1) input preprocessing, 2) model inference, and 3) post-processing operations across heterogeneous computing units (CPU, GPU, and NPU) to ensure high throughput while maintaining low latency.

This pipeline efficiently handles each camera frame by performing only minimal operations (like bitmap conversion and queueing) on the main thread, then delegating heavier tasks to specialized thread pools. Player detection and ball tracking each follow a clear flow—preprocess on the `PreprocessThreadPool`, then run inference on dedicated single-thread executors, which prevents race condition as `tfliteInterpreter` is not thread-safe. Postprocessing writes results into a thread-safe map, and a final retrieval step checks frames in sequence to dispatch complete data to the UI. By cleanly separating tasks and carefully managing concurrency, this design supports straightforward scalability for future expansions such as pose estimation and court tracking.

![program_architecture](.\Asset\program_architecture.png)

Based on the architecture workflow given above, here's a summary:

1. **Camera → analyze()**: For every frame, do minimal synchronous work:

2. Convert to Bitmap.
3. Create `FrameRes`, store in `resultsMap`.
4. Possibly submit `PlayerDetTask` or skip.
5. Buffer the frame for ball tracking; if 3 frames available, submit `BallTrackingTask`.
6. Close `ImageProxy`.

7. **Preprocessing**: In `PlayerDetTask` or `BallTrackingTask` → run on `PreprocessThreadPool`.

8. **Inference**: Submits a `PrioritizedTask` to either `playerExecutor` or `tennisExecutor`.

9. **Postprocess**: Once the inference completes on that single-threaded executor, store bounding boxes / ball positions in `resultsMap`.

10. **Retrieve Result**: Check if the next frame in sequence is complete. If yes, remove from the map, notify UI, compute FPS, etc.



### TrackNet Knowledge Distillation

Initially, we attempted `transfer learning` by fine-tuning the pretrained TrackNetv2 model (designed for tennis ball detection) on our dataset. Despite achieving high accuracy, the model’s large size (~11M parameters) rendered it unsuitable for mobile deployment, even after full integer quantization. To address this, we adopted `knowledge distillation` to transfer TrackNetv2’s capabilities to a compact student model. The student model combines a `ResNet18` backbone with transposed convolution layers, reducing parameters by 5x,  and achieving **90% precision/recall** while maintaining real-time performance on edge devices. Here's a [video](https://youtu.be/GjFr1I6eo_Q) which demonstrate the student model trained from scratch vs. trained with knowledge distillation.