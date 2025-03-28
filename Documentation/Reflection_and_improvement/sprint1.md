## Table of Contents

- [Table of Contents](#table-of-contents)
- [Complete Tasks](#complete-tasks)
  - [Player Detection On Android](#player-detection-on-android)
  - [TrackNet Knowledge Distillation](#tracknet-knowledge-distillation)
- [Reflection on quantizing tennis tracking model TrackNet v2](#reflection-on-quantizing-tennis-tracking-model-tracknet-v2)
  - [The good parts](#the-good-parts)
  - [The challenges](#the-challenges)
  - [Actionable improvement](#actionable-improvement)
- [Reflection on Trial of using TrackNet v4 to replace with the current ball tracking model](#reflection-on-trial-of-using-tracknet-v4-to-replace-with-the-current-ball-tracking-model)
  - [The good parts](#the-good-parts-1)
  - [Challenges](#challenges)
  - [Actionable improvement](#actionable-improvement-1)
- [Reflection on successes](#reflection-on-successes)
- [Area of Improvements](#area-of-improvements)
- [Key lessons learned during the sprint](#key-lessons-learned-during-the-sprint)
- [Actionable improvements](#actionable-improvements)
- [Task estimations and velocity tracking](#task-estimations-and-velocity-tracking)
- [Evaluation based on reflection](#evaluation-based-on-reflection)

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
- The TrackNet v4 is more lightweight thhan TrackNet v2.

### Challenges
- In some scenarios, the model does not perform very well. There is not a significant improvement compared with TrackNet v2. There is an example showing in the [test video](https://github.com/Darcyddx/Active-Vision/blob/main/Documentation/Reflection_and_improvement/Asset/test_video1.mp4).

### Actionable improvement
- We use the distilled model from ResNet_TrackNet v2 and get better performances. There is an example showing in the [test video](https://github.com/Darcyddx/Active-Vision/blob/main/Documentation/Reflection_and_improvement/Asset/test_video2.mp4).



`The following is the reflection for the whole sprint 1 period`

## Reflection on successes
Our project had many great achievements. A clear and well-structured Product Backlog was established, with tasks categorized by priority to ensure alignment with project goals. This prioritization was complemented by adopting an appropriate task estimation method, which significantly enhanced our team’s understanding of task complexity. Additionally, leveraging Github Projects as an agile tool proved invaluable, as it seamlessly integrated code, documentation, tasks, and status updates into a cohesive workflow. This not only streamlined communication but also provided full transparency across all stages of product development.

Another key area of success was user requirements analysis. Six detailed user stories were crafted, each representing different user groups, including beginners, coaches, and advanced players. These user stories were meticulously designed with clear acceptance criteria, ensuring that feature completeness could be easily verified during development. Furthermore, function priorities were strategically set to focus on core features first. For instance, ball tracking was prioritized for initial deployment on mobile devices, followed by player detection in Sprint 1. This phased approach ensured that critical functionalities were delivered promptly while maintaining flexibility for future iterations.

On the technical front, the project benefited from a carefully selected technology stack tailored to the project's needs. Android was chosen as the primary platform, with adaptive devices used for testing to ensure compatibility across various hardware configurations. To enhance model performance, multiple backup strategies were implemented, such as model distillation and the decision to upgrade from TrackNetV2 to TrackNetV4. Mobile optimization was another critical consideration, with efforts focused on leveraging NPU acceleration and TensorFlow Lite to improve efficiency and reduce latency on mobile devices.

## Area of Improvements 

While the project achieved significant milestones, there are several areas where improvements can be made to enhance overall effectiveness. One notable area is documentation timeliness. Although documentation was maintained throughout the project, there were instances where updates lagged behind the actual progress. To address this, it is essential to establish a more disciplined approach to ensure that all documents are updated in detail and in real-time as the project evolves. This will not only improve clarity but also serve as a reliable reference for future iterations or handovers.

Task dependencies also present an opportunity for improvement. While interdependencies among tasks were acknowledged, their descriptions were often too vague, leading to potential confusion during execution. Providing more detailed explanations of these dependencies would enable better task prioritization and resource allocation. For example, creating a dependency matrix or visual flowchart could help the team understand how individual tasks contribute to broader objectives, thereby fostering a more cohesive workflow.

Risk assessment, although addressed through the establishment of a risk management mechanism, could benefit from more quantifiable evaluations. While risks were identified, the specific impacts of some risks were not thoroughly assessed. To strengthen this process, it is recommended to introduce specific monitoring indicators and early warning mechanisms for each identified risk. Additionally, developing more robust tackling strategies—such as contingency plans or fallback options—would further mitigate potential disruptions.

Finally, progress tracking requires refinement to ensure that milestones are met consistently and transparently. Currently, the lack of a detailed progress tracking mechanism has led to occasional gaps in accountability. Introducing milestone checkpoints at regular intervals, along with standardized progress report templates, would provide a clearer picture of the project’s trajectory. This would not only facilitate timely identification of bottlenecks but also allow for proactive adjustments to keep the project on track.

By addressing these areas for improvement while building on the successes achieved, the team can further enhance its ability to deliver high-quality results efficiently and effectively.
The combination of strategic planning, user-centric design, and technical innovation has been instrumental in driving the project forward. However, continuous refinement in documentation, task management, risk assessment, and progress tracking will be crucial to sustaining momentum and achieving long-term success.

## Key lessons learned during the sprint 
In this short iteration (Sprint), our team completed various core functions from player detection, tennis ball tracking, and replacing TrackNet v4 with TrackNet v2. By using quantitative models and enabling hardware acceleration such as NPU, we have achieved stable and impressive real-time detection results on the mobile side and also maintained high compatibility across multiple models. Here are the key lessons. 

First, we realized that cross-team collaboration and clarity of requirements were essential to accelerate the project schedule. In this project, we need to complete different functional modules such as player detection, site identification and UI improvement as soon as possible, which need to do data interface and dependency management as soon as possible, otherwise it will cause repeated development or resource conflict. Secondly, in terms of quantitative models and hardware acceleration, although the inference delay has been successfully reduced, the efficiency and accuracy of loading on the mobile side is still a huge challenge. We recognize the need for adequate compatibility testing and timely update of the fallback logic during actual deployment. 

In addition, balancing accuracy and efficiency is always a key problem. The TrackNet v4 model does have advantages in reasoning speed, but the accuracy improvement is not obvious in individual scenarios. We try to combine techniques such as knowledge distillation and mixing accuracy to achieve better detection results in the end. However, the model can still be affected by background interference from spectators, billboards, etc., in more complex real game scenes. This made us aware of the inadequacy of our data set, which, despite the large amount of match video we have available through platforms such as YouTube, still does not fully cover the diverse interference environment. This made us more aware of the advantages of diverse datasets, such as the urgent need to mark and collect scenes with dense audiences or chaotic backgrounds to improve the accuracy and robustness of the model in real competition environments during subsequent training and fine-tuning phases. 

In summary, Sprint successfully implemented several key features, but we also deeply recognized the potential challenges in mobile deployment, precision control and hardware adaptation. We will continue to learn and improve the overall performance of the application. 

## Actionable improvements 
First of all, in our AI tennis tracking project, the TrackNet model has demonstrated effective tennis ball detection and trajectory prediction by using deep learning-based heatmap analysis. However, the system currently suffers from a high false-positive (FP) rate, where non-ball objects (e.g., shadows, spectators, or camera artifacts) are frequently misclassified as the tennis ball. This introduces significant noise into the tracking pipeline. To address this issue, we would apply Kalman Filters or Optical Flow to refine TrackNet’s outputs. These methods use motion patterns to filter out illogical detections. For example, if a "ball" suddenly appears far from its expected path or remains static, the algorithm flags it as noise and predicts the ball’s true position. Such method might reduce false alarms while keeping processing fast for real-time use. 

In addition, we have integrated YOLOv8 on mobile devices for real-time human detection, prioritizing speed and accuracy. While I implemented frame-skipping mechanism to reduce computational load, efficiency remains limited on mobile devices. To optimize, we will integrate lightweight tracking algorithms (ByteTrack/SORT) to propagate detections across skipped frames. The tracking algorithm will "remember" detected people across skipped frames, so YOLOv8 doesn’t need to check every frame. As a result, including tracking algorithms would benefit real-time performance on edge devices without sacrificing robustness. 

Last but not least, while our human detection (using YOLOv8) works, it lacks robustness. Tennis courts often include irrelevant people (spectators, ball kids, referees). To filter out non-players, we aim to retain only athletes by checking if their positions are within the court boundaries. This requires detecting the court’s corner coordinates. However, current open-source court detection algorithms perform poorly. We noticed similarities between court boundary detection and autonomous driving’s parking slot detection (used to identify parking space markings). We plan to adapt these proven parking-slot detection methods to accurately locate tennis court corners and boundaries. Once implemented, this will enable precise athlete filtering by mapping detected humans to court-specific zones. 

## Task estimations and velocity tracking 

In the first sprint of our project, the Burn-down Chart served as a critical tool for visualizing the completion of tasks and managing the project’s progress. This chart displayed a trend of remaining tasks throughout the sprint, which was intended to decrease as tasks were systematically completed. Initially, our team exhibited a commendable effort, effectively reducing the number of open tasks of coding module. This was in line with our planned velocity and task estimations, suggesting a strong start and alignment with our expected timelines. 

However, a pivotal moment arose mid-sprint, where the chart showed an unexpected increase in the number of remaining tasks. This spike indicated either the addition of new tasks due to evolving project requirements or the identification of previously unforeseen challenges that necessitated immediate attention. This deviation was a critical insight; it prompted us to review our initial task estimations and the team’s velocity. We realized that our initial estimates were too optimistic or did not fully account for the complexity involved in some of the tasks. As a result, we conducted a thorough review session with the team to identify the root causes of these discrepancies. 

From this reflection, several adjustments were made to better align our future sprint planning with the actual team performance observed. Firstly, task estimations were adjusted to be more realistic, considering the detailed feedback from team members about the time and resources required to complete specific tasks. We also refined our velocity tracking method to incorporate a buffer for unexpected developments, thus allowing more flexibility in our project management approach. 

Additionally, this spike in the Burn-down Chart facilitated a deeper discussion about our team’s workflow and resource allocation. It became evident that better communication and more frequent check-ins could prevent similar issues in future sprints. We decided to implement daily stand-up meetings to ensure all team members were aligned and any potential blockers were addressed promptly. 

In conclusion, the insights gained from the Burn-down Chart during Sprint 1 were invaluable. They not only highlighted the need for more accurate task estimations and velocity measurements but also fostered a culture of continuous improvement and adaptability within our team. Going forward, these adjustments are expected to enhance our project management practices and lead to more efficient sprint executions, ultimately ensuring that we meet our project milestones in a timely and effective manner. 

## Evaluation based on reflection

First, training the YOLOv8 model for human detection on human-related datasets like COCO and CrowdHuman, handled by Kerry, was harder than expected. We estimated it at 6 story points, thinking these datasets would work well for tennis players. But during training, we found that the model struggled with tennis-specific cases, like players in weird poses or holding rackets. After reflection, we decided to fine-tune the model by adding some tennis match clips to the dataset. For example, we grabbed 50 short videos from online matches and labeled them ourselves. This took an three extra days, but it made the model better at spotting players on the court. Next time, we’ll mix in sports data from the start instead of relying only on general datasets like COCO. 

Second, quantizing the YOLOv8 model was tougher than we thought. Zhiyuan Lu worked on it, and it was estimated at 8 story points because it’s a big task. The model worked fine on a computer, but on mobile devices, it was too slow. Sometimes taking 2 seconds per frame. We reflected during a meeting with clients and agreed to spend extra time optimizing it with NPU acceleration. Tao Lu helped with this, even though his optimization task was planned for later. We tested it on a real phone and got the speed down to under 1 second per frame. This change was worth it, but it pushed our testing phase back by two days. We learned we need to test on real devices earlier, not just simulators. 

The Android app camera feed integration, led by Xi Ding, went smoothly at first. It was set at 5 story points, and we got the camera working in the app fast. But when Peiling Lam added the detected player bounding boxes to the display (4 story points), we noticed a problem. The boxes flickered on the screen because the model output wasn’t syncing well with the camera feed. After communicating with each other, we added a small fix to stabilize the output, which took extra few days. This wasn’t in our original plan, so we realized we underestimated the time for linking the model to the app. For Sprint 2, we’ll add buffer time for these kinds of tasks. 

We changed our training process by adding tennis data, started optimization earlier, and fixed display issues on the spot. These shifts improved our results, even if we need to reschedule our timeline a bit. But these adjustments made our work stronger. Moving forward, we’ll plan more time for real-device testing, double-check our estimates for app integration, and tackle performance issues from the beginning. Reflection taught us to be flexible and catch problems early, which will help us in the next sprints. 

