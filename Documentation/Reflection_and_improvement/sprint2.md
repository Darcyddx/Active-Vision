## Table of Contents

- [Table of Contents](#table-of-contents)
- [Complete Tasks](#complete-tasks)
  - [Player Keypoints Detection/Pose Estimation On Android](#Player Keypoints Detection/Pose Estimation On Android)
  - [Tennis Swing Recognition on Android](#Tennis Swing Recognition on Android)
- [Successes from Sprint 1](#Successes from Sprint 1)
  - [Challenges Faced](#Challenges Faced)
  - [Areas for Improvement](#Areas for Improvement)
  - [Feedback from Sprint 1 Review](#Feedback from Sprint 1 Review)
  - [Actions Based on Feedback](Actions Based on Feedback)
  - [Reflection on Agile Methodology and Weekly Client Feedback](Reflection on Agile Methodology and Weekly Client Feedback)
- [Player Keypoints Detection/Pose Estimation On Android](#Player Keypoints Detection/Pose Estimation On Android)
  - [The good parts](#the-good-parts-1)
  - [Challenges](#challenges-1)
  - [Actionable improvement](#actionable-improvement-1)
- [Tennis Swing Recognition on Android](#Tennis Swing Recognition on Android)
  - [The good parts](#the-good-parts-2)
  - [Challenges](#challenges-2)
  - [Actionable improvement](#actionable-improvement-1)
- [Reflection on successes](#reflection-on-successes)
- [Area of Improvements](#area-of-improvements)
- [Key lessons learned during the sprint](#key-lessons-learned-during-the-sprint)
- [Actionable improvements](#actionable-improvements)
- [Task estimations and velocity tracking](#task-estimations-and-velocity-tracking)
- [Evaluation based on reflection](#Evaluation based on reflection)



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

## Reflection on successes


## Area of Improvements


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


## Task estimations and velocity tracking


## Evaluation based on reflection

