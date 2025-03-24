### User Story 1
#### Basic Information
**ID:** USER-001  
**Title:** Android Interface for Tennis App  
**Priority:** Low  
**Milestone:** Sprint 3

#### User Story Description
**As an** Android phone user,  
**I want to** have an Android-compatible interface,  
**so that** I can conveniently use the app on my device.

#### Persona Description

- education and experience
  - graduate university student majoring in Finance (Bachelor degree).
  - has a general familiarity with mobile applications and technology but may not have technical expertise in software development. 
  - comfortable navigating mobile apps and expect a smooth user experience.
- relevant interest
  - interested in tennis and sports analytics, likely using the app for tracking performance, analyzing gameplay, or improving skills. 
  - value convenience, accessibility, and a seamless experience on their Android device.
- job
  - work as a accountant in one of the Big4 companies.
- personalization
  - prefer an intuitive interface with clear visuals, easy navigation, and real-time feedback.

#### Acceptance Criteria
1. **Scenario 1:** Accessing the app on an Android device  
   - **Given** I have an Android phone with the app installed,  
   - **When** I launch the app,  
   - **Then** I see a fully functional interface optimized for Android.


#### Additional Information
- **Related Epic:** Mobile App Accessibility  
- **Dependencies:** None
- **Technical Notes:** Ensure compatibility with Android device NPU.  
- **Notes:** Test on multiple even old Android devices for consistency.

---

### User Story 2
#### Basic Information
**ID:** USER-002  
**Title:** Ball Trajectory and Player Posture Detection for Beginners  
**Priority:** High  
**Milestone:** Sprint 1

#### User Story Description
**As a** tennis beginner,  
**I want to** see the ball’s trajectory and have my posture and swing detected,  
**so that** I can quickly learn tennis with basic detection features.

#### Persona Description

- education and experience
  - first year university student studying Bachelor of Genetics.
  - a beginner in tennis with little to no formal training. 
  - have limited experience with sports technology and analytics but are comfortable using mobile apps.
- relevant interest
  - passionate about learning tennis efficiently and improving their technique.
  - interested in using technology to accelerate their progress and make training more engaging.
- job
  - student, has heavy workload on assignments and labs.
  - have limited time for in-person coaching and rely on digital tools for self-improvement.
- personalization
  - prefers a straightforward, beginner-friendly interface with minimal setup required.

#### Acceptance Criteria
1. **Scenario 1:** Viewing ball trajectory  
   - **Given** I am recording a practice session with the app,  
   - **When** I hit a ball,  
   - **Then** the app displays the ball’s path overlaid on the video.
2. **Scenario 2:** Detecting player position  
   - **Given** I am recording a practice session with the app,  
   - **When** I move in the court,  
   - **Then** the app sends back a bounding box of my position.
3. **Scenario 3:** Detecting player posture  
   - **Given** I am in a recorded session,  
   - **When** I perform a swing,  
   - **Then** the app highlights my posture and suggests corrections.

#### Additional Information
- **Related Epic:** Beginner Training Features  
- **Dependencies:** US-001 (Android Interface)
- **Technical Notes:** Requires integration with device camera and basic ML model for the detections.  
- **Notes:** Focus on simplicity for beginners; avoid overwhelming with too many functions.

---

### User Story 3
#### Basic Information
**ID:** USER-003  
**Title:** Ball Position Detection Relative to Court  
**Priority:** Medium  
**Milestone:** Sprint 3

#### User Story Description
**As a** tennis coach,  
**I want to** detect the ball’s position relative to the court, to help me make specific plan which are more targeted for tennis players
**so that** I can easily see whether the ball lands in or out of bounds.

#### Persona Description

- **education and experience*
  - Graduated from the School of Physical Education or Master's Degree in Sports Management
  - Have rich experience in competition and team guidance
  - Aspiring to train outstanding tennis players and promote tennis sports development
- relevant interest
  - Interested in detailed analytics such as ball speed, trajectory angles, and other performance metrics.
- job
  - provide professionl advice for professional tennis players to prepare competitions
  - tennis Scout(by analysing tennis match recording to find potential talented players
- personalization
  - Prefers an intuitive user interface with robust video analysis features that include professional insights.

#### Acceptance Criteria
1. **Scenario 1:** Recorded Match Analysis
   - **Given**  I am reviewing footage from a recent training session or match,
   - **When** the ball lands during play,
   - **Then** the app should accurately annotate the ball’s landing position on an interactive court diagram with labels such as "In" or "Out."

2. **Scenario 2:** Live Practice Feedback
   - **Given** I am coaching during a live session,
   - **When** a ball is struck and lands on the court,
   - **Then**  the app should immediately provide a visual indicator on the court diagram to show the ball’s landing zone.

#### Additional Information
- **Related Epic:** Coaching Tools 
- **Dependencies:** Requires integration with the Android Interface US-001 (Android Interface)
- **Technical Notes:**  Implementation requires precise court boundary recognition and might require manual calibration to account for variations in court dimensions and color differences in court surfaces.
- **Notes:** Coaches may need an export feature for session analysis (future story).
- **Future Improvement:** Consider adding an export feature for detailed session analysis and player development tracking.

---

### User Story 4
#### Basic Information
**ID:** USER-004  
**Title:** Ball Speed Calculation for Advanced Players  
**Priority:** Medium 
**Milestone:** Sprint 3

#### User Story Description
**As an** advanced tennis player,  
**I want to** calculate the speed of the ball,  
**so that** I can further improve my skills.

#### Persona Description

- *education and experience*
  - Played tennis from middle school through college, with a focus on sports science and biomechanics.
  - 8+ years of competitive tennis experience in regional and amateur tournaments.
  - Frequently applies data analytics to improve technique.

- relevant interest
  - Interested in advanced player stats—ball speed, spin rates, and shot angles.
  - Follows professional tennis circuit closely for training insights.
- job
  - Tennis coach at a local sports academy.
  - Aspires to compete at higher-level tournaments while balancing coaching responsibilities.
- personalization
  - Motivated by challenging personal records; sets speed goals for each session.
  - Enjoys testing new technologies (apps, trackers, sensors) that offer detailed performance metrics.

#### Acceptance Criteria
1. **Scenario 1:** Measuring ball speed in practice  
   - **Given** I am recording a shot with the app,  
   - **When** I hit the ball,  
   - **Then** the app displays the ball’s speed in km/h or mph.

2. **Scenario 2:** Reviewing speed trends  
   - **Given** I have recorded multiple shots,  
   - **When** I access my session stats,  
   - **Then** the app shows an average speed and highlights my fastest shot.

#### Additional Information
- **Related Epic:** Advanced Player Performance  
- **Dependencies:** US-002 (Ball Trajectory Detection)  
- **Technical Notes:** Speed calculation requires frame-by-frame analysis; ensure accuracy within ±5 km/h.  
- **Notes:** Option to toggle between metric and imperial units.

---
### User Story 5
#### Basic Information
**ID:** USER-005
**Title:** Support for Singles and Doubles Tennis Matches on Android
**Priority:** Medium
**Milestone:** Sprint 2

#### User Story Description
**As an** tennis match lover,  
**I want to** use an Android-compatible interface that allows me to select both singles and doubles match types,  
**so that** I can conveniently enjoy different tennis match formats on my device.

#### Persona Description

- **education and experience*

- relevant interest
- job
- personalization

#### Acceptance Criteria
1. **Scenario 1:** Choosing singles or doubles match type  
   - **Given** I am on the match setup screen,  
   - **When** I select a match type,  
   - **Then** I can choose between a singles match or a doubles match and proceed accordingly.

### User Story 6

#### Basic Information

**ID:** USER-006
**Title:** Tennis Club Manager
**Priority:** Low
**Milestone:** 

#### User Story Description

**As an** tennis club manager,  
**I want to** use an integrated tennis analysis program,  
**so that** I can give my VIPs to record their matches and provide them with cool effects and analysis for their games.

#### Persona Description

- **education and experience*

- relevant interest
- job
- personalization

#### Acceptance Criteria

1. **Scenario 1:** Choos  
   - **Given** I a,  
   - **When** I ,  
   - **Then** I can .

