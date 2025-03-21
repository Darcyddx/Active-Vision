# Meeting Notes

**Date:** 6 Mar 2025  
**Time:** 4 PM – 5:30 PM  
**Location:** Zoom online meeting  
**Attendees:** Arjun Raj (Client), All Team Members, Dr Lei Wang  
**Recorder:** Pei Ling Lam  

## Agenda Items

### 1. Explain the Android app codes to Dr Lei Wang and how it works

### 2. Discuss about the focus in this semester with Dr Lei Wang

- **Main ball detection** – multiple balls on the same tennis court
- **Open pose for court detection** – Open pose is only for human keypoints instead of court keypoints.
  - The main challenge is the inability to detect the court if some parts are covered. Hence, we need to capture the whole court as the condition.
- **Check the equations to calculate the speed of the ball**
- **About finetuning TrackNetv4**
  - It is lighter than v3 but heavier than v2, and it provides better accuracy.
- **Court detection** – check existing model
- **Challenges when dealing with machine learning techniques on the Android app**
  - Android Studio doesn’t have many machine learning libraries like numpy and needs to implement them from scratch.

### 3. Focus in 25s1

- **Robustness and lightweight of the models**
- **Main ball tracking** instead of other captured balls and calculating the speed.
- **Challenging scenarios** such as illumination.
- **Multiple players and pose estimation.**