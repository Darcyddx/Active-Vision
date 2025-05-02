# Sprint Plan and Task Estimation


## 1. Product Backlog

| Feature                              | Priority | Notes                                                        |
| ------------------------------------ | -------- | ------------------------------------------------------------ |
| Player detection                     | High     | Core feature to track relevant players                       |
| Court detection                      | High     | To show player's location and ball dropping points on a mini-map |
| Player pose estimation               | High     | Analyse player's technical movements                         |
| Calculate tennis ball speed          | Medium   | Analyse ball speed                                           |
| Replace TrackNetv2 with TrackNetv4   | Medium   | Make the model lightweight and increase performance          |
| Improve app user interface           | Low      | User's can easily navigate the APP                           |
| Improve robustness of existing model | Low      | Improve overall performance including speed of inference and accuracy |

## 2. Estimation and Velocity Tracking

### Sprint 1

Each task is estimated based on its complexity and dependencies. Time is allocated in story points (SP) where:

Small (S): 1-3 SP (Easy, minimal dependencies)
Medium (M): 4-6 SP (Some dependencies, moderate effort)
Large (L): 7-10 SP (Complex, critical dependencies)

![Burn-down graph for sprint1](https://github.com/user-attachments/assets/4e7177ce-edba-40ab-839e-30481183fa5f)

During Sprint 1 of our project, we utilized a Burn-down Chart to closely monitor the progression of tasks and ensure timely completion. This tool depicted the number of remaining tasks over the course of the sprint, ideally demonstrating a downward trajectory as tasks were completed. Initially, we observed a steady decrease in task count, indicating a productive start. However, mid-sprint, there was a noticeable increase in remaining tasks, suggesting the introduction of new tasks or unexpected challenges that were not initially accounted for. Towards the end of the sprint, a sharp decline followed by a slight rise in the task count was evident, highlighting a significant push by the team to wrap up tasks, though some last-minute additions or incomplete tasks slightly altered the expected end trajectory. This Burn-down Chart proved essential for maintaining an overview of project status and dynamically adjusting our strategies to address any arising challenges effectively.

![Velocity_Chart](https://github.com/user-attachments/assets/3f259167-2dfa-48a5-8416-f763942c2ac9)

The velocity chart for "Active Vision" provides insights into the team's performance across three iterations, highlighting both planned and completed story points. In Iteration 1 (W03), the team planned to complete 2 story points but only managed to finish 1, indicating a significant gap between expectations and actual delivery. 

Moving to Iteration 2 (W04), the team planned to complete 19 story points but delivered only 10, which represents an improvement in terms of total output compared to Iteration 1 but still shows a notable shortfall. This discrepancy might indicate that the team is gradually increasing its capacity but is still facing challenges in accurately estimating or executing tasks.

In Iteration 3 (W04), the team planned to complete 26 story points and successfully delivered 18, showing continued growth in productivity. Although there is still a gap between planned and done, the team has demonstrated consistent progress over the iterations, suggesting improved planning accuracy and execution efficiency. However, the persistent gap highlights areas where further refinement in estimation or process optimization may be needed.

Overall, the chart reveals a positive trend in the team's ability to deliver more work over time, but it also underscores the need for ongoing improvements in planning accuracy and addressing any bottlenecks that prevent full completion of planned work. Continuous monitoring and adaptation will be crucial to achieving higher alignment between planned and actual velocities.

### Sprint 2

![output (2)](https://github.com/user-attachments/assets/d67154aa-83e7-43fc-8752-fd4cad8c5c81)
Starting from April 23, the Burn-down Chart for Sprint 2 reveals significant project dynamics. On April 23, the number of remaining tasks began to rise sharply—from 8 to 18 by April 29. This upward trend suggests that new tasks were introduced mid-sprint or that earlier underestimated tasks were redefined with added complexity. The lack of steady task completion during this period highlights challenges in planning accuracy and potential overcommitment.

A dramatic shift occurred on April 30, where the remaining tasks dropped from 18 to 10 in one day, indicating a concentrated effort by the team to close out deliverables before the sprint's end. Despite this progress, the final count on May 1 shows that 10 tasks remained open, signaling a gap between our planned velocity and actual delivery capacity.

This pattern underscores the importance of continuous task review, realistic estimation, and more granular mid-sprint tracking. Moving forward, we aim to improve workload balancing and velocity prediction to ensure a smoother and more predictable sprint outcome.


![Sprint2](https://github.com/user-attachments/assets/d6774145-5c34-4d54-ae38-984adf65bcc6)

The velocity chart for **"Active Vision"** outlines the progression of planned versus completed (done) story points across four iterations: *Before Mid-term*, *Week07*, *Week08*, and *Week09*. This visualization helps assess the team’s planning accuracy and delivery performance over time.

In the **Before Mid-term** phase, the team planned 26 story points but only completed 18. This initial gap reflects early-stage challenges, potentially due to underestimating task complexity or overcommitting. It sets a baseline for measuring future improvement.

During **Week07**, planned work increased to 30 points, with 22 story points completed. This iteration marks a clear productivity improvement—both in total completed work and in narrowing the gap between planned and done. The increase in completed story points may indicate better familiarity with project scope and improved team coordination.

In **Week08**, planning rose significantly to 39 story points, yet only 23 were delivered. While there is a slight improvement in actual velocity compared to the previous week, the growing gap suggests either over-ambitious planning or potential execution bottlenecks that limited output.

By **Week09**, the team planned the highest amount of work—46 story points—and delivered 36. This is the best performance across all iterations, both in terms of absolute completed work and the ratio of planned-to-done (nearly 78%). It suggests that the team has matured in its sprint planning and execution capacity.

Overall, there is a steady upward trend in both planned and completed story points, indicating improving productivity. The gap between planned and completed work decreases over time, especially in Week09, which shows strong planning-execution alignment. The team should continue refining estimation practices, ensuring workload matches realistic capacity, and identifying early blockers that may limit progress.

Would you like this formatted into a slide or report section?

## 3. Sprint Goals/Milestones

### Sprint 1 - Player detection + Enhance TrackNet Performance
| Task	                                                                | Estimation	  | Dependencies	         | Task allocation     |
|----------------------------------------------------------------------|--------------|------------------------|---------------------|
| Train YOLOv8 for human detection on human-related dataset(COCO, CrowdHuman...) | 6 SP (M)      | Model training      | Kerry               |
| Quantize YOLOv8 model for player detection and deploy the model on Android	| 8 SP (L)      | Model integration | Zhiyuan Lu          |
| Implement Android app camera feed integration                         | 5 SP (M)      | App framework setup     | Xi Ding             |
| Display detected player bounding boxes on the app, co-work with the TrackNet	| 4 SP (M)      | Model integration       | Peiling Lam         |
| Construct a lightweight segmentation network(student model) and perform knowledge distillation on trained TrackNetV2(teacher model) | 6 SP (M)      | Model enhancement | Yichi Zhang         |
| Optimize real-time processing (NPU acceleration, TensorFlow Lite)	 | 7 SP (L)      | Model deployment        | Tao Lu              |

**Sprint goal**: Develop player detection with real-time tracking in Android app


### Sprint 2 - Player pose estimation
| Task	                                                              | Estimation	 | Dependencies	    | Task allocation     |
|--------------------------------------------------------------------|-------------|------------------| ---------------------|
| Integrate OpenPose/MediaPipe for pose estimation                   | 	7 SP       | Player detection | Zhiyuan Lu |
| Extract and classify tennis postures (forehand, backhand)	         | 8 SP        | Pose model setup | Peiling Lam, Yichi Zhang |
| Overlay key posture points on the app UI                           | 	5 SP       | Pose estimation  | Xi Ding  |
| Optimize inference time to maintain real-time processing           | 	6 SP       | Model tuning     | Kerry    |
| Android Interface for Tennis App | 4 SP | -  | Tao Lu |

**Sprint goal**: Detect human poses and classify tennis movements

### Sprint 3 - Court detection
| Task	                                                               | Estimation	 | Dependencies	        |
|---------------------------------------------------------------------|-------------|----------------------|
| Label dataset for court keypoints detection                         | 	3 SP       | Dataset availability |
| Train different models on court dataset to test which one is better | 	7 SP       | Model training       |
| Implement court keypoints detection for Android app                 | 	9 SP       | Model integration    |
| Replace TrackNetV2 with TrackNetV4 for better ball tracking         | 	8 SP       | -                    |
| Compute real-time ball speed estimation                             | 	3 SP       | Ball tracking        |
**Sprint goal**: Detect court keypoints, implement ball speed calculation

## 4. Sprint Backlog

(We will use this function directly from GitHub repo -> Projects -> Team planning -> [Backlog](https://github.com/users/Darcyddx/projects/1))

## 5. Tools and Management Methods

Clearly describe agile tools and methodologies used:

- **Agile Methods:** Daily Stand-ups, Sprint Reviews, Sprint Retrospectives
- **Management Tools:** GitHub Projects
- **Version Control:** Git
- **Communication:** Discord, WeChat

## 6. Risk and Issue Management

Clearly outline potential risks, their impacts, and mitigation strategies:

| Risk Description                                    | Potential Impact                                        | Mitigation Plan                                              | Owner             |
| --------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ | ----------------- |
| **Low accuracy of player recognition model**        | Users lose confidence, leading to poor app adoption.    | Early evaluation with different models; collect high-quality data to retrain frequently. | ML Engineer       |
| **Insufficient labeled dataset for training**       | Poor model performance and delays in project timelines. | Establish labeling pipeline; use external datasets; crowdsourcing to expedite labeling. | Data Engineer     |
| **Integration issues between frontend and backend** | Delayed release, lower app stability and reliability.   | Regular integration meetings, implement CI/CD for rapid issue identification and resolution. | Backend Developer |
| **Technology or framework incompatibility**         | Technical debt, rework, delayed timelines.              | Early technology stack validation; prototyping before major implementation decisions. | Technical Lead    |
| **Performance bottlenecks on mobile devices**       | Reduced user experience due to slow app performance.    | Regular performance testing; optimization strategies (e.g., lightweight models, inference optimization). | Mobile Developer  |

------

**Note:** Continuously update the documentation throughout the sprint to reflect progress accurately.
