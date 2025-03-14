# Meeting Minutes

**Date:** March 8, 2025  
**Time:** 2:30 PM - 3:30 PM  
**Location:** Online Teams Meeting  
**Recorder:** Xi Ding  
**Attendees:** Everyone in the team and clients  

## Agenda

- Clarify the focus of Active Vision project for S1 2025
- List prioritized tasks
- Assign responsibilities and deadlines

---

## Meeting Notes

### 1. Clarify the focus of Active Vision project for S1 2025

The meeting began with a discussion on the core focus areas for the S1 project:

- **Robustness:** Enhance system stability under diverse conditions.
- **Ball Tracking:** Improve accuracy and reliability of ball detection and tracking.
- **Multiple Players:** Support scenarios with multiple players (1v1 and 2v2 matches).
- **Challenging Scenarios:** Address complex situations such as multi-ball environments and varying court scenarios.

---

### 2. List Prioritized Tasks

The team outlined the following prioritized tasks to guide development efforts over the semester.  

#### 2.1 Improve Robustness of Ball Tracking, Player Detection, and Pose Estimation

**1.1 Implement Tennis Court Keypoints Detection Algorithm**  
- **Objective:** Develop an algorithm to detect key points on the tennis court (e.g., lines).  
- **Purpose:** Provide spatial context for player and ball tracking.  

**1.2 Use Tennis Court Keypoints to Filter Players and Support 1v1 and 2v2 Matches**  
- **Objective:** Leverage keypoints to distinguish players and adapt the system for singles (1v1) and doubles (2v2) matches.  
- **Purpose:** Improve player detection accuracy in multi-player scenarios.  

**1.3 Use Tennis Court Keypoints to Track the Main Ball in Multi-Ball Scenarios**  
- **Objective:** Filter out secondary balls and track the primary ball using court context.  
- **Purpose:** Enhance robustness in challenging multi-ball environments.  

**1.4 Replace TrackNetV2 with TrackNetV4 for Better Ball Tracking**  
- **Objective:** Upgrade the existing ball tracking model to TrackNetV4.  
- **Purpose:** Achieve higher accuracy and performance in ball detection.  

#### 2.2 Perform Data Analytics and Provide a User-Friendly Interface

**2.1 Calculate Ball Speed**  
- **Objective:** Develop a module to compute and display ball speed in real-time.  
- **Purpose:** Provide actionable insights for users.  

**2.2 Visualize Ball Dropping Points and Player Movements**  
- **Objective:** Create a graphical representation of ball trajectories and player positions.  
- **Purpose:** Enhance data interpretability for end-users.  

**2.3 Improve the User Interface for Better Usability**  
- **Objective:** Design the UI based on user feedback and usability testing.  
- **Purpose:** Ensure an intuitive and efficient user experience.  

#### 2.3 Improve the Efficiency of the Whole Model

**3.1 Replace the Current Model if a More Efficient Model is Found**  
- **Objective:** Continuously evaluate and integrate superior models as they become available.  
- **Purpose:** Maintain cutting-edge performance throughout the project lifecycle.  

#### 2.4 Analyze Player Playing Strategies

**4.1 Classify and Analyze Player Shots (e.g., Backhand, Forehand)**  
- **Objective:** Build a system to categorize and analyze different types of player shots.  
- **Purpose:** Provide strategic insights for training and performance analysis.  

---

### 3. Assign Responsibilities and Deadlines

For **Week 3**, the team decided to focus on **Human Pose Recognition**.  

A follow-up meeting will be scheduled in the **first week of Sprint 1** to distribute responsibilities on prioritized tasks (**March 14, 2025**).
```

这个Markdown格式的文档可以直接用于GitHub、Notion或其他Markdown支持的工具。你可以复制粘贴到Markdown编辑器中查看效果！需要进一步调整的话，告诉我哦！ 😊