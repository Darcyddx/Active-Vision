## Table of Contents

- [Complete Tasks](#complete-tasks)
  - [Ball Speed Calculation on Android](#ball-speed-calculation-on-android)
  - [Court Detection](#court-detection)
- [Reflection on Team Communication and Problem Solving](#reflection-on-team-communication-and-problem-solving)
- [Reflection on Successes](#reflection-on-successes)
- [Key Lessons Learned During the Sprint](#key-lessons-learned-during-the-sprint)
- [Actionable Improvements](#actionable-improvements)
- [Task Estimations and Velocity Tracking](#task-estimations-and-velocity-tracking)
- [Evaluation Based on Reflection](#evaluation-based-on-reflection)

## Complete Tasks

### Ball Speed Calculation on Android

####  **Implementation Explanation**

The method `calculateSpeed()` calculates ball speed by analyzing recent position data across a fixed number of frames (10 frames in this case):

##### **Steps in the Method:**

1. **Historical Data Extraction**
   - It uses the last 10 frames stored in the `history` list.
   - Calculates the difference (`dx`, `dy`) between the first and last frame positions and the elapsed time (`dt`) between these frames.
2. **Pixel-to-Meter Conversion**
   - Converts the distance from pixels to meters using the predefined constant `PIXEL_TO_METER_RATIO`.
3. **Speed Calculation**
   - Computes speed in meters per second (MPS) using the distance and elapsed time.
   - Converts MPS to kilometers per hour (KMH).
4. **Threshold Checking**
   - Checks if the calculated speed meets the minimum speed threshold (`MIN_SPEED_KMH_THRESHOLD`), otherwise disregards the measurement.
5. **Unit Conversion and Return**
   - Returns the calculated speed according to the specified speed unit (KMH, MPH, or MPS).

##### **Formula Used:**



Speed (m/s) = distance (pixels) / (PIXEL_TO_METER_RATIO × time interval (s))

------

#### **Good Parts**

1. **Flexible Unit Selection**
   - Supports multiple units (KMH, MPH, MPS), enabling flexibility and regional applicability.
2. **Use of Multiple Frames**
   - Uses multiple frames (10) rather than just 2-3 frames, improving reliability and smoothing noise in speed calculation.
3. **Pixel-to-Meter Calibration**
   - Includes explicit conversion (`PIXEL_TO_METER_RATIO`) allowing real-world measurements, crucial for accuracy in practical applications.
4. **Detailed Debug Logging**
   - Provides detailed logs with calculation steps, aiding debugging and system fine-tuning.

------

#### **Challenges**

1. **Accuracy of Pixel-to-Meter Ratio**
   - The fixed `PIXEL_TO_METER_RATIO` assumes a consistent camera perspective and distance. If camera angles or distances vary, speed calculations can become inaccurate.
2. **Frame Rate Dependency**
   - The accuracy depends heavily on stable frame rates. Irregular frame rates or dropped frames can significantly affect speed accuracy.
3. **Rigid Speed Threshold**
   - The fixed speed threshold (`MIN_SPEED_KMH_THRESHOLD`) may not be suitable for varying conditions or different sports scenarios, potentially causing misclassifications.

------

#### **Actionable Improvements**

##### 1. **Adaptive Pixel-to-Meter Calibration**

- Implement automatic calibration based on known reference points within the frame (e.g., court markings, net height)

##### 2. **Robust Frame Rate Handling**

- Introduce logic to detect and handle abnormal frame intervals, discarding unreliable calculations if frame rates fluctuate significantly

##### 3. **Dynamic Speed Threshold**

- Adjust threshold dynamically based on historical speed data or statistics to enhance accuracy and context relevance

##### 4. **Enhanced Directional Analysis**

- Extend current calculations to include directional vectors, improving shot type inference and trajectory analysis

### Court Detection

#### **Implementation Explanation**

The `CourtDetector` class is designed for detecting a tennis court or similar sports courts from images using an ONNX deep learning model. The main workflow includes:

**1. Initialization and Model Loading**

- Loads an ONNX model (`.onnx`) using `OrtEnvironment` and initializes a session (`OrtSession`).
- Reads the model bytes from Android’s assets folder.

**2. Preprocessing**

- Accepts an Android Bitmap image, resizes it to a fixed input size (`640x640`).
- Converts the bitmap into a 4-dimensional float tensor (`[1, 3, 640, 640]`), representing channels RGB separately, to prepare input data for ONNX Runtime inference.

**3. Inference**

- Creates an `OnnxTensor` from the preprocessed image data.
- Runs inference via ONNX Runtime’s session, passing the image tensor into the model and retrieving outputs from multiple specified layers.

**4. Post-processing**

- Extracts the relevant output (keypoints, in this scenario) from inference results.
- Converts ONNX outputs into usable Java structures (`float[][][]`).

------

#### **Good Parts**

- **Clear Separation of Concerns**
  - Explicit functions for model loading, preprocessing, inference, and post-processing enhance readability and modularity.
- **ONNX Runtime Integration**
  - Efficiently utilizes ONNX Runtime for model inference, enabling performant execution.
- **Asset Management**
  - Properly reads model files from Android assets, suitable for mobile app deployment.

------

#### **Challenges (Issues Observed)**

You’re currently facing multiple critical runtime issues:

##### **1. Segmentation Fault (SIGSEGV)**

- **Cause**: Passing null or incorrectly managed tensors into native ONNX Runtime.
- **Context**: Happens specifically during `OrtSession.run(...)`.

##### **2. FP16 Not Supported Error**

- **Cause**: Model contains FP16 precision operations incompatible with your device's SoC.
- **Context**: ONNX model optimized for FP16, not supported on your Android hardware.

##### **3. Delegate Initialization Errors**

- **Cause**: Attempting to use GPU/NPU delegates that are unsupported or not configured correctly on your device.

------

#### **Actionable Improvements (Step-by-Step Solutions)**

##### **1. Fixing the Segmentation Fault**

- **Check for Null References**:
   Before inference, verify all objects:
- **Verify Input Names & Dimensions**:
   Confirm input/output tensor names (`"image"`, `"output"`, etc.) match your ONNX model exactly.
- **Session Thread-Safety**:
   Ensure your inference sessions are properly synchronized if using multiple threads. Don’t close sessions during inference.

------

##### **2. Address FP16 Issue (Hardware Compatibility)**

- **Re-export or Convert the Model without FP16**:
   Use standard precision (FP32) during model export/conversion:
- **ONNX Runtime Session Options**:
   Explicitly disable FP16 if available during initialization:

------

#### **3. Resolve Delegate Initialization Errors**

- **Fallback to CPU Execution (recommended)**:
   Run inference without GPU/NPU delegates to ensure basic functionality first:
- **If GPU/NPU needed**:
  - Verify delegate compatibility with your hardware.
  - Ensure proper delegate `.so` libraries and permissions (like DSP) are available and configured correctly.

------

#### **4. Improve Robustness and Error Handling**

- Add detailed error logs and guards to prevent runtime crashes:

- Use try-catch comprehensively

## Reflection on Team Communication and Problem Solving

### Specific event

During this sprint, I encountered a significant technical challenge while attempting to deploy our Detectron2 model to TensorFlow Lite (TFLite). The intended workflow involved converting the model first to ONNX, then to TensorFlow, and finally to TFLite. However, due to the unique architecture of Detectron2, I was only able to export it successfully to ONNX opset 16—and that too after making some custom modifications to the Detectron2 export scripts.

The real obstacle began when attempting to convert the ONNX model to TFLite. The onnx2tf tool failed due to unimplemented operations, and the onnx-tf library only supports opset versions up to 12, making it incompatible with our exported model. Despite trying several other conversion tools and approaches, every attempt to reach TFLite deployment failed.

At that point, I turned to my team and raised the issue in our communication channel. By discussing the problem openly, we collectively evaluated alternatives and made a pragmatic decision to keep the model in ONNX format and explore deploying it directly to the mobile application. Surprisingly, the ONNX model worked well in real-time performance tests on the app, ultimately fulfilling our core requirement.

This experience was an important reminder that open communication and team collaboration are essential, especially when facing complex technical roadblocks. Initially, I had been trying to solve the problem on my own, but reaching out allowed us to spot overlooked options and arrive at a viable solution more efficiently. It highlighted the value of diverse perspectives in problem-solving—something that’s easy to miss when working in isolation.

### Lessons Learned

- Don’t hesitate to ask for help: Discussing technical issues with the team early can uncover alternative solutions and save time.
- Be flexible with goals: While TFLite deployment was ideal, ONNX served as an effective fallback with acceptable performance.
- Communication channels are strategic tools: They’re not just for updates, but for brainstorming and decision-making under uncertainty.

### Actionable Improvements for the Future

- Set up short technical syncs or debugging huddles when someone is blocked, to accelerate resolution.
- Encourage early reporting of blockers in daily or weekly check-ins.
- Maintain a shared document of known limitations and alternatives for model deployment strategies to reduce trial-and-error in future projects.

## Reflection on successes

In Sprint 3, the project made significant strides in functionality by introducing **ball speed calculation** and **court detection** into the mobile application. These features expand the system’s analytical capabilities and pave the way for richer user feedback and performance insights.

The ball speed module was successfully integrated with support for multiple units (KMH, MPH, MPS) and demonstrated consistent, smooth performance by analyzing 10-frame motion windows. This increased the robustness of our speed estimates while maintaining real-time viability. The pixel-to-meter calibration—although currently static—provided a solid baseline for contextualizing in-game events.

In parallel, court detection via an ONNX model marked a key milestone in spatial reasoning for the app. The separation of preprocessing, inference, and post-processing tasks led to a clean, modular implementation suitable for real-world extension.

Importantly, when conversion of Detectron2 models to TFLite proved unfeasible, the team collaboratively pivoted to deploying the ONNX model directly. This pragmatic shift preserved core functionality and demonstrated flexibility in adapting to technical constraints, reinforcing the team’s growing maturity in decision-making.

## Key lessons learned during the sprint 

Sprint 3 provided us with valuable insights into the importance of adaptability, technical depth, and effective team communication, especially when dealing with complex model deployment and system integration challenges. One of the most important lessons was the need to have contingency plans in place when facing toolchain limitations or architectural constraints. Initially, we aimed to convert a Detectron2 model into TensorFlow Lite for mobile inference. However, the process was hampered by significant compatibility issues—such as ONNX opset mismatches and unsupported TFLite operations. Progress stalled until we openly discussed the issue within the team and collectively decided to switch to direct ONNX deployment using ONNX Runtime. This “second-best” solution ultimately performed well in real-time scenarios, proving that pragmatism and flexibility often outweigh perfection in real-world development.

We also learned that hardware compatibility checks must be addressed early in the model deployment pipeline. For instance, our FP16-optimized model ran into device-specific issues, especially on mid-range Android devices that lacked full support for half-precision computation. These failures—such as segmentation faults during inference—highlighted the need to validate model compatibility on actual target hardware early in the workflow, rather than relying solely on emulator or desktop tests. Moving forward, we plan to incorporate device profiling and fallback mechanisms into our deployment strategy to minimize risk and debugging overhead.

Another key takeaway from this sprint was the strategic importance of team communication in overcoming technical roadblocks. Initially, the model conversion problem was tackled in isolation, leading to wasted time. But once the issue was raised in a group discussion, the team quickly identified a viable alternative (ONNX deployment) and implemented it successfully. This reinforced the idea that technical discussions are not just about reporting progress, but also about surfacing blockers early and leveraging collective insight. We now plan to introduce more frequent technical syncs and impromptu “debugging huddles” when anyone on the team encounters significant issues.

From an architectural perspective, Sprint 3 also emphasized the value of modular and scalable design. Whether it was the multi-frame analysis approach used in ball speed calculation, or the structured preprocessing-inference-postprocessing pipeline in the court detection module, the benefits of clear component separation were evident. These design patterns made debugging easier, accelerated development, and laid a strong foundation for future feature enhancements. It reminded us that maintainability and extendibility should be top priorities when designing any system component.

Sprint 3 significantly enhanced our capabilities in model inference and system integration while also strengthening our project management, risk handling, and team collaboration skills. These lessons will serve as a solid foundation for the next development phase and bring us closer to building a robust, flexible, and user-friendly mobile application.

## Actionable improvements

Following the technical breakthroughs and challenges of Sprint 3, several actionable improvements have been identified to ensure more robust functionality, better performance, and smoother user experience in future iterations. First and foremost, we plan to implement a more dynamic and adaptive pixel-to-meter calibration mechanism. Currently, the system relies on a fixed pixel-to-meter ratio, which assumes a static camera angle and consistent field of view. This limits the accuracy of real-world measurements such as ball speed. To improve upon this, we aim to automatically calibrate the scale using known reference objects in the frame—such as court lines or the net—allowing for real-time spatial adaptation across different recording setups and devices.

In parallel, we recognized the need for a more intelligent and context-aware thresholding mechanism in our ball speed analysis. The existing hardcoded speed threshold, used to filter out unrealistic or irrelevant detections, lacks flexibility and fails to adapt to different scenarios. To address this, we will develop a dynamic thresholding system that learns from historical speed data or adjusts based on match context, ensuring better balance between sensitivity and precision.

Another critical area for improvement is runtime hardware compatibility. Our current model export pipeline inadvertently included FP16 operations, which caused runtime failures on devices lacking proper hardware support. Moving forward, we will incorporate a hardware capability check at app startup and ensure the system can automatically fall back to CPU inference when GPU/NPU delegates are unavailable or unstable. This guarantees baseline functionality even under constrained hardware conditions.

Additionally, we plan to build a deployment reference guide that documents our experience with ONNX, TFLite, and other formats—highlighting known limitations, conversion bottlenecks, and compatibility insights. This guide will serve as a shared knowledge base for the team and accelerate future model deployment efforts by reducing redundant trial-and-error cycles.

Finally, in light of the segmentation faults and initialization errors we encountered, we will improve error logging and exception handling throughout the model inference pipeline. Comprehensive try-catch blocks, detailed log outputs, and graceful fallback procedures will enhance the app’s robustness and help diagnose issues more efficiently during testing and in the field.

These improvements represent a blend of technical refinement, system resilience, and workflow maturity, all of which are essential as we move closer to deploying a production-grade mobile application capable of delivering reliable real-time tennis analytics.

## Task estimations and velocity tracking

In Sprint 3, our team demonstrated incremental improvement in task execution discipline, but several gaps in estimation accuracy and velocity tracking remain evident. As shown in the burn-down chart, the sprint began with a consistent number of 10 open tasks that remained unchanged during the first few days (May 1–May 4), suggesting either delays in task kick-off or overestimation of early sprint velocity.

A sudden drop to 6 tasks on May 5 indicates a burst of progress, likely triggered by focused effort or resolution of earlier blockers. However, the temporary increase to 7 open tasks on May 9 reveals that a previously completed task may have been reopened or that new scope was introduced mid-sprint. This event underscores the need for more robust definitions of “done” and better risk anticipation during estimation.

During the mid-sprint period (May 9–May 20), the chart reflects stagnation, with little change in remaining task count. This flatline suggests that some tasks were underestimated in complexity, or that resource allocation did not align with planned velocity. Despite this, a final drop on May 22 to just 2 remaining tasks shows the team’s ability to mobilize toward sprint closure, though the concentration of work near the deadline indicates a reactive rather than steady delivery pace.

Going forward, we aim to refine our estimation strategy by incorporating historical velocity trends and using smaller, more measurable sub-tasks. We also plan to increase the frequency of velocity reviews during sprints and improve workload balancing. These changes will help align planned velocity with actual team capacity, leading to more predictable delivery and fewer mid-sprint surprises.

## Evaluation based on reflection
Sprint 3 marked a significant milestone for our team, showcasing growth in technical expertise, problem-solving, and collaboration. Reflecting on our work reveals what went well, where we faced challenges, and how we can improve moving forward. This evaluation, written in clear and simple language, captures our progress authentically.

Our team made notable strides in Sprint 3 compared to earlier sprints. In Sprint 1, tasks often took longer than expected due to underestimated complexity. Sprint 2 brought delays from task dependencies. By Sprint 3, we managed challenges more effectively through open communication and practical solutions. For example, when converting our Detectron2 model to TensorFlow Lite (TFLite) failed due to compatibility issues, we quickly pivoted to deploying the ONNX model directly. This adaptability kept us on track and ensured we delivered key features, demonstrating improved risk management and problem-solving.

The ability to adjust when plans faltered, like with the TFLite export, highlighted our growth as a team. We delivered meaningful features, such as ball speed calculation and court detection, which made our app smarter and more game-aware. These additions laid a strong foundation for real-time tennis analytics. Team communication also became more proactive, enabling faster resolution of blockers compared to earlier sprints, where issues lingered longer.

Despite these successes, challenges remain. Static pixel-to-meter calibration limited the accuracy of ball speed calculations when camera angles varied. Device compatibility issues, such as unsupported FP16 operations, caused crashes on some Android devices. Task progress was uneven, with much of the work completed close to the sprint’s end, suggesting room for better planning.

Sprint 3 was a turning point. We overcame tough technical hurdles, delivered impactful features, and grew as a collaborative unit. Compared to Sprint 1’s overruns and Sprint 2’s bottlenecks, this sprint showed stronger coordination and better fallback planning. These experiences taught us to plan more realistically, communicate openly, and adapt to setbacks. By addressing ongoing challenges, like calibration and hardware compatibility, we can make future sprints smoother and more efficient.

Our team is now better prepared to handle real-world deployment constraints. We’ve developed a more practical approach to feature planning, task estimation, and user experience validation. The lessons from Sprint 3 will guide us to work smarter and deliver a high-quality, user-friendly app in the next phase of development.
