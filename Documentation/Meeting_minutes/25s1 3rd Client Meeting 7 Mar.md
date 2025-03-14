# Meeting Details

- **Date:** 7 Mar 2025
- **Time:** 2:15 PM – 3:15 PM
- **Location:** Zoom
- **Attendees:** All Team Members, Arjun Raj (Client), Francis (Client), Dr Lei Wang
- **Recorder:** Pei Ling Lam

## Agenda Items

1. **Explain the Android app codes to Francis and how it works**

2. **Work Done**
   - Ball tracking
   - Human detection
   - Human pose estimation
   - Basic working app

3. **Semester Goal**
   - App interface improvement
   - Easy deployability – TFLite for quantization
     - One click deployment
   - Court keypoints detection – to detect who are the players playing
   - How to handle multiple players
     - Different pose and playing strategies (backhand / forehand / serving)
   - Different scenarios - colours of tennis court
   - Robustness on existing model
   - Add ball speed, ball dropping point on mini court

4. **Task Prioritization (ordered)**
   - Tennis court keypoints – use existing one
     - Remove some layers of the model to lightweight the model for deployment
     - Use transfer learning for lightweight model
     - Try OpenCV harris corner detection
   - Try TrackNetv4 and lighweight the model, add ball speed and ball dropping point on mini court
   - Different pose and playing strategies (backhand / forehand / serving)
   - Improve user interface, one click deployment
   - Try replacing the existing model with better model, model distillation

5. **Send emails for weekly updates**