# Meeting Details

- **Date:** 21 Mar 2025
- **Time:** 2 PM – 2:40 PM
- **Location:** Zoom
- **Attendees:** All Team Members, Arjun Raj (Client)
- **Recorder:** Xi Ding

## Agenda Items

1. **Work Done**
   - Research on DMPR-PS for keypoints detection
     - if we want to use the label tool from DMPR-PS itself to label our tennis court dataset, something should be modified. In the research, there are two kinds of label: marks and slot.
     - Fine-tuning strategy
     - Sprint 3: Court detection
   - Model distillation from TrackNet V2 to ResNet_Track
   - Test TrackNet V4 and explore whether it can be applied in our system
     - Environment issues

2. **Task Allocation**
   - Persona (each user distributed to each team member)
   - Meeting minutes of tutorial and clients(Xi Ding)
   - Player detection (Sprint 1)
     - Collect and preprocess dataset for player detection (Kerry)
     - Quantize YOLOv8 model for player detection (Zhiyuan Lu)
     - Implement Android app camera feed integration (Xi Ding)
     - Display detected player bounding boxes on the app (Peiling Lam)
     - Test model performance on real-world tennis matches (Yichi Zhang)
     - Optimize real-time processing (NPU acceleration, TensorFlow Lite) (Tao Lu)

3. **Client Feedback**
    - Consider the trade-off between speed and precision for model quantization.
    - Test player detection with non-professional players to ensure generalizability.