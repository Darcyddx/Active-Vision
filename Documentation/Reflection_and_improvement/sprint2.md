## Table of Contents

- [Table of Contents](#table-of-contents)
- [Complete Tasks](#complete-tasks)
  - [Player Keypoints Detection Pose Estimation On Android](#player-keypoints-detection-pose-estimation-on-android)
  - [Tennis Swing Recognition on Android](#tennis-swing-recognition-on-android)
- [Successes from Sprint 1](#successes-from-sprint-1)
  - [Challenges Faced](#challenges-faced)
  - [Areas for Improvement](#areas-for-improvement)
  - [Feedback from Sprint 1 Review](#feedback-from-sprint-1-review)
  - [Actions Based on Feedback](#actions-based-on-feedback)
  - [Reflection on Agile Methodology and Weekly Client Feedback](#reflection-on-agile-methodology-and-weekly-client-feedback)
- [Player Keypoints Detection Pose Estimation On Android](#player-keypoints-detection-pose-estimation-on-android)
  - [The good parts](#the-good-parts-1)
  - [Challenges](#challenges-1)
  - [Actionable improvement](#actionable-improvement-1)
- [Tennis Swing Recognition on Android](#tennis-swing-recognition-on-android)
  - [The good parts](#the-good-parts-2)
  - [Challenges](#challenges-2)
  - [Actionable improvement](#actionable-improvement-2)
- [Reflection on successes](#reflection-on-successes)
- [Area of Improvements](#area-of-improvements)
- [Key lessons learned during the sprint](#key-lessons-learned-during-the-sprint)
- [Actionable improvements](#actionable-improvements)
- [Task estimations and velocity tracking](#task-estimations-and-velocity-tracking)
- [Evaluation based on reflection](#evaluation-based-on-reflection)



## Successes from Sprint 1

- Successfully integrated OpenPose/MediaPipe for pose estimation within the Android app, following the successful completion of player detection in Sprint 1.
- Completed classification of tennis postures into forehand and backhand, providing foundational structure for feedback features.
- The app UI now overlays keypoints for detected poses, improving the visual feedback mechanism.
- Real-time inference speed has been significantly improved through targeted model optimization and tuning, maintaining responsiveness during testing.

### Challenges Faced

- Adapting OpenPose output to mobile format required custom data formatting and additional pre/post-processing.
- Initial inference times exceeded our real-time requirement, requiring iterative profiling and tuning.
- Classification accuracy for postures was inconsistent during dynamic motion, indicating a need for temporal smoothing or sequential modeling.

### Areas for Improvement

- Improve robustness of posture classification by integrating temporal features or applying smoothing techniques over a sequence of frames.
- Begin logging false detections or misclassifications systematically for targeted model refinement.
- Enhance UI clarity for non-technical users (e.g., highlight pose keypoints with context-aware labels).

### Feedback from Sprint 1 Review

No specific areas for improvement were highlighted by the tutor or stakeholders. This positive feedback reinforces the current approach, but we will continue to actively seek more specific, constructive input going forward.

### Actions Based on Feedback

Since no corrective feedback was received, we focused on:
- Maintaining the same workflow and division of labor.
- Enhancing model performance and UX as natural progression from last sprint’s success.
- Continuing documentation and feedback tracking rigorously to prepare for future feedback cycles.

### Reflection on Agile Methodology and Weekly Client Feedback

The use of the Agile methodology in our project—structured into three distinct sprints—has significantly contributed to the clarity, organization, and continuous progress of our development process. Each sprint was defined with a specific milestone and goal, ensuring the team remained focused and that deliverables were aligned with stakeholder expectations. This sprint-based structure also allowed us to divide complex tasks into smaller, manageable components, which improved our ability to prioritize and allocate resources effectively.

One of the most valuable aspects of our Agile process has been the weekly meetings with our client. These regular check-ins have created a consistent feedback loop, enabling us to:

- Demonstrate incremental progress,
- Clarify requirements in real time,
- Address misunderstandings early,
- And build trust through transparency and accountability.

Although feedback in some weeks was minimal, the presence of a structured channel for client interaction ensured alignment with the project vision and goals. These meetings also helped us remain adaptive—able to pivot or improve based on the client’s evolving input.

In summary, Agile's sprint-based structure and ongoing collaboration with the client fostered a focused, iterative, and responsive development cycle. It allowed us to maintain momentum while staying aligned with both technical goals and user expectations.

## Player Keypoints Detection Pose Estimation on Android

### The good parts 1
* **Light-weight model choice.** Replacing MoveNet with **MobileNetV2-based keypoint detection** cut the TFLite file to ~4.3 MB after FP16 quantisation, keeping memory overhead low for mid-tier devices.  
* **Real-time FPS on device.** With the GPU delegate enabled we measured ~25 FPS on a Pixel 6 and ~18 FPS on a Galaxy A54—well above the 15 FPS target for smooth overlays.  
* **Clean Android abstraction.** The new `PlayerPoseEstimator` helper wraps interpreter initialisation, output parsing, and drawing utilities, keeping the CameraX pipeline uncluttered.  
* **Dataset alignment.** Sprint 2 finished annotating and trimming the 13-joint format that downstream models expect, unifying data from the public TennisFrames set and our own recordings.  

### Challenges 1
* **Keypoint jitter & ID swaps.** Because MobileNetV2 outputs single-frame predictions, fast motions occasionally swap wrists/elbows between frames, propagating noise into the RNN.  
* **Device fragmentation.** Older phones without a GPU/NPU delegate throttle to <10 FPS; keeping both CPU and GPU paths doubled maintenance work.  
* **Conversion quirks.** Some custom ops were exported as *SELECT_TF_OPS*, inflating the binary until we rewrote them as pure TFLite ops.  

### Actionable improvement 1
1. **Temporal smoothing.** Add a one-euro filter or exponential moving average on joint positions before feeding the RNN to suppress jitter.  
2. **Lazy delegate selection.** Detect available NNAPI/GPU delegates at runtime and fall back gracefully, logging FPS for real-world stats.  
3. **Quantisation-aware retraining.** Re-train MobileNetV2 with INT8 QAT to push FPS even higher on low-end hardware while preserving accuracy.  

---

## Tennis Swing Recognition on Android

### The good parts 2
* **End-to-end pipeline working on-device.** We now stream 30 × 13 × 2 keypoints into a sliding window and feed a **bidirectional GRU-based RNN** (`tennis_rnn.tflite`), producing 4-class soft-max probabilities every frame (Serve, Forehand, Backhand, Neutral).  
* **Unified tensor shape.** Standardising on `[1 × 30 × 26]` simplifies both Java and native pipelines and matches the training script.  
* **Visual feedback.** Action labels and per-class confidence bars render in the live preview, immediately exposing mis-classifications during field testing.  

### Challenges 2
* **Cold-start latency.** The first inference after interpreter creation takes ~250 ms because of model allocation and delegate initialisation—noticeable when the user opens the camera.  
* **Class imbalance.** Serve samples are <6 % of the training set, so recall is only 0.61 versus 0.88–0.92 for groundstrokes.  
* **Sequential buffering.** Maintaining a 30-frame ring buffer per player thread-safely while keeping GC pressure low required several refactors.  

### Actionable improvement 2
1. **Pre-warm interpreters.** Move TFLite interpreter creation into the Application class and keep a warmed-up singleton to hide cold-start delays.  
2. **Data augmentation.** Generate synthetic Serve sequences (speed variation, horizontal flips, temporal warping) to balance the dataset and raise Serve recall above 0.8.  
3. **Early-exit strategy.** Investigate frame-wise confidence accumulation so obvious actions can exit the window early, reducing average inference cost from 30 to ~20 frames.  

---



## Reflection on successes
1. **Real-time Pose Estimation and Swing Recognition on Android:**  
   Sprint 2 marked a significant milestone in terms of technical integration and system-level coordination. We successfully deployed *real-time player keypoint detection and tennis swing recognition* on the Android platform.

2. **Lightweight Keypoint Detection with MobileNetV2:**  
   The on-device deployment of a MobileNetV2-based keypoint detection model was completed. After FP16 quantization, the TFLite model size was reduced to ~4.3MB, minimizing memory usage. With GPU delegate enabled, the model achieved ~25 FPS on Pixel 6 and ~18 FPS on Galaxy A54—well above the 15 FPS target for smooth overlay.

3. **Modular CameraX Integration:**  
   A new `PlayerPoseEstimator` class was developed to encapsulate model loading, output parsing, and drawing. This kept the CameraX pipeline clean and modular, facilitating future maintenance and scalability.

4. **Unified Data Format for Keypoints:**  
   We standardized the 13-joint keypoint format across the public TennisFrames dataset and our own recordings, ensuring compatibility with downstream GRU models. This improved data quality, alignment, and training consistency.

5. **End-to-End GRU-based Swing Classification:**  
   A bidirectional GRU model was deployed to classify swing types (Serve, Forehand, Backhand, Neutral) using 30-frame keypoint sequences. The unified tensor shape `[1 × 30 × 26]` simplified integration between Java and native pipelines and matched training scripts.

6. **Visual Feedback in Real Time:**  
   Action labels and confidence bars were rendered live in the app UI, providing clear and intuitive feedback. This transparency helped expose misclassifications and guided testing in the field.

7. **Responsiveness to Technical Challenges:**  
   We addressed cold-start latency by introducing a *pre-warmed interpreter* strategy. For class imbalance—especially the underrepresented Serve class—we planned synthetic data augmentation for future model refinement.

8. **Prioritizing Real-Time Performance:**  
   This sprint confirmed that *real-time performance cannot be an afterthought*. From model selection and input resolution to hardware acceleration, every design choice was guided by speed and responsiveness.

9. **Scalability Across Devices:**  
   To support low-end hardware, we began implementing compatibility mechanisms like *deferred delegate binding* and *runtime fallback between NNAPI and CPU*, ensuring smooth performance across a range of Android devices.

10. **Foundation for Future Iterations:**  
    Sprint 2 successfully integrated the pose and action recognition systems into the Android pipeline. These features are now functional, testable, and visually informative—laying the groundwork for refinement, usability testing, and extended features in Sprint 3.



## Key lessons learned during the sprint 
1.  **Integrating External Libraries Needs Careful Planning:**
    *   Adding tools like OpenPose or MediaPipe wasn't just a coding task. We spent significant time setting up the necessary configurations, making sure the versions were compatible with our existing project, and figuring out how to correctly feed our player detection data into the pose estimation model.
    *   There were unexpected small issues, like specific installation steps and understanding the exact input format the library needed. This "setup" and "learning curve" took up a noticeable chunk of the 7 SP estimated for integration.
    *   **Lesson:** For future sprints involving new, complex libraries, we should specifically budget time not just for *using* the library, but also for *installing, configuring, and learning* its basics. Estimations should account for potential setup hurdles.

2.  **Defining Specific Poses is Complex and Iterative:**
    *   Trying to classify "forehand" vs. "backhand" using just the skeleton points (pose estimation output) proved quite challenging. We realized that people perform these poses differently – variations in style, speed, and even camera angle made it hard to create simple rules (like "if elbow angle is X, it's a forehand").
    *   We likely spent time trying different rules, testing them on various video clips, and finding they didn't always work. The 8 SP for this task reflects this difficulty.
    *   **Lesson:** Simple rules might not be enough for reliable pose classification. We need clearer definitions, possibly more examples (data) of each pose, or maybe even explore slightly more advanced classification techniques later. Getting this right requires testing with lots of different examples early on.

3.  **Real-Time Performance is a Major Hurdle:**
    *   Getting the pose estimation model to run fast enough for a smooth, real-time experience was a significant challenge, as reflected in the 6 SP dedicated to optimization. We found that running the complex pose models on every single video frame could slow down the app considerably or use too much battery.
    *   We had to investigate *why* it was slow (e.g., model size, image resolution) and experiment with solutions like using a simpler model, reducing the input video size, or finding smarter ways to process the frames.
    *   **Lesson:** Performance for real-time features cannot be an afterthought. We must consider the processing speed right from the beginning when choosing models and designing the workflow. Optimization isn't just a final polish; it's a core part of making real-time feedback usable. We need to keep monitoring performance as we add more features.

4.  **Dependencies Between Tasks Can Cause Bottlenecks:**
    *   We saw how tightly connected our tasks were. For example, work on overlaying posture points and developing the feedback system couldn't really start properly until the basic pose estimation was working reasonably well.
    *   If the pose estimation was inaccurate or delayed, it directly impacted multiple other tasks.
    *   **Lesson:** We need to be very mindful of these dependencies when planning sprints. If a critical task like pose estimation is proving difficult, it puts other related tasks at risk. We should communicate blockers clearly and perhaps think about building temporary data or simpler versions so other work can continue in parallel where possible.

5.  **Translating Pose Data into Useful Feedback Requires Thought:**
    *   Simply drawing the skeleton points on the screen is one thing, but turning that information into helpful coaching advice is another level of complexity. We learned that just knowing the joint positions isn't enough.
    *   We need to figure out *what* the user needs to know and how to translate the raw angles and positions from the pose model into that simple, actionable advice. This required thinking about the *meaning* behind the pose, not just the data itself.
    *   **Lesson:** Building an effective feedback system requires more than just technical skill. It needs an understanding of tennis technique and careful consideration of how to present information clearly to the user. We should probably involve user testing or expert input early to make sure the feedback we plan to give is actually helpful and easy to understand.

## Actionable improvements

During Sprint 2, our team achieved significant milestones in developing a real-time human pose estimation and tactical action recognition system for mobile deployment. We successfully delivered keypoint detection and pose classification tasks on schedule, deploying the models to mobile devices. However, several challenges and areas for improvement emerged during this phase.

**Keypoint Detection**
We optimized the MobileNetV2-based keypoint detection model by quantizing it to FP16 precision, enabling stable 30 FPS inference on mobile GPUs. While this demonstrated feasibility for real-time use, attempts to further reduce memory and power consumption by quantizing to INT8 resulted in significant accuracy degradation. Given that FP16 consumes more memory and increases GPU power usage, resolving INT8 quantization issues will be critical for long-term scalability and user experience. To enhance performance, we plan to explore more efficient architectures like LiteHRNet and RTMPose, which have shown superior accuracy-efficiency trade-offs in recent research.

**Pose Classification**
For tactical action recognition, we implemented an RNN-based model that processes keypoint sequences to classify athlete movements. While functional, the model’s accuracy remains suboptimal, likely due to limited training data and overfitting. To address this, we will prioritize expanding our dataset with diverse athlete scenarios and augmenting existing data. Additionally, we aim to experiment with state-of-the-art spatial-temporal graph convolutional networks (ST-GCN), which could better capture skeletal joint relationships and motion patterns.

**Team Collaboration & Workflow**
Our teamwork improved markedly compared to previous terms, particularly in stakeholder communication, task prioritization, and cross-functional coordination. However, GitHub branch management remains a pain point. Frequent merge conflicts and inconsistent branching strategies consumed substantial time during integration. Moving forward, we will adopt a standardized Git workflow (e.g., GitFlow) and enforce stricter code review practices to streamline collaboration. Tools like automated CI/CD pipelines and branch protection rules may also help mitigate these issues.

**Next Steps**
Immediate priorities include resolving INT8 quantization challenges, benchmarking LiteHRNet/RTMPose for mobile compatibility, and expanding the pose classification dataset. For team processes, refining version control practices will be essential to sustain productivity as the project grows in complexity. While hurdles remain, Sprint 2 demonstrated our ability to deliver functional solutions under tight deadlines while identifying actionable paths for technical and operational refinement.

This phase underscored the importance of balancing optimization with model robustness and highlighted the need for scalable engineering practices as we progress toward a production-ready system.


## Task estimations and velocity tracking
During Sprint 2, our team's task estimations and velocity tracking revealed both progress and areas for improvement. From the Burn-down Chart, we observed that remaining tasks initially stayed constant before experiencing a sudden increase starting April 24, rising from 8 to 18 tasks by April 29. This indicated either underestimated task complexity or new tasks being added mid-sprint. The subsequent drop on April 30—from 18 to 10—shows a last-minute push to close tasks, further suggesting initial estimations lacked accuracy and workload was not evenly distributed throughout the sprint.

In parallel, the velocity chart demonstrated that while 26 story points were planned for the sprint, only 18 were completed. This 69% completion rate indicates a shortfall in meeting planned capacity. Compared to prior iterations, the planned workload increased significantly, but actual delivery did not grow proportionally. This discrepancy suggests overambitious planning without fully accounting for task dependencies, development bottlenecks, or testing overhead.

The mismatch between planned and completed work highlights the need to refine our estimation practices. For example, complex research or integration tasks may require more detailed breakdown and buffer time, while dependency-heavy tasks should be flagged during sprint planning. Additionally, the velocity tracking mechanism should be adjusted to factor in actual team throughput, using historical velocity as a baseline instead of relying solely on task count or perceived difficulty.

Going forward, we plan to adopt more granular task slicing, incorporate estimation techniques like planning poker, and continue tracking actual versus estimated effort through story points. By aligning planning efforts with real performance trends, we aim to improve predictability and sprint execution efficiency.

## Evaluation based on reflection

1. **What we did Well:**
Using agile and sprints worked: Breaking the project into smaller tasks made the work clear. We knew what to do each time. The weekly client meetings were useful, even when feedback was small. It kept us on the right track and helped us show our progress step by step. This made us feel organized.
We built core features: We successfully added pose estimation (seeing the player's keypoints) and started sorting poses like forehand and backhand. Getting the keypoints to show up on the screen in the app was a good visual success. We also made the app run faster, which was a big challenge.
We identified problems clearly: When things were hard, like the pose points jumping around and the app being slow at first, we figured out why. The reflections on challenges for keypoint detection and swing recognition were detailed.

2. **What was difficult and what we learned:**
Real-time is hard: Making the app work smoothly in real-time on phones was tougher than expected. We learned that choosing the right model and optimizing it (like using the GPU) is super important right from the start. We also saw that different phones perform differently.
Details matter: We learned that small things can take time. Setting up libraries like OpenPose/MediaPipe wasn't just plug-and-play; it needed careful setup. Getting the data format right between different parts (like pose estimation and the RNN) took effort. Defining poses accurately (forehand vs. backhand) was also complex because people move differently.
Need for better data/techniques: We saw that our pose classification wasn't perfect, especially during fast movements or for less common actions. This showed us we need better ways to handle movement over time and maybe need more or better training data.

3. **How reflection helped Us improve:**
Led to clear actions: The best thing about reflecting was turning problems into specific "Actionable Improvements." For example, noticing the keypoint jitter led directly to planning temporal smoothing. Seeing the serve classification was weak led to planning data augmentation. Realizing the app started slow led to the idea of pre-warming the model.
Better planning for future: Understanding that setup takes time, or that real-time performance is tricky, helps us estimate tasks better for the future. We know now to budget time for optimization and dealing with different devices.

4. **Overall:**
This reflection shows we made real progress but also faced typical challenges in building this kind of app. We learned valuable lessons about performance, data, and the complexity of seemingly simple tasks. Because we reflected and identified specific ways to improve, we feel more prepared and focused for the next steps in making the app more accurate and user-friendly.