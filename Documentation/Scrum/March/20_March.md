# Daily - stand ups 20/03/25

### Pei Ling Lam
## What have I done?
- Ball tracking algorithm for mobile app in java
- play around the code of DMPR-PS

## What am I doing next?
- Continue to write ball tracking algorithm 
- Try to use their label tool and see how they label on their dataset, then label on tennis court dataset if possible 

## Challenges 
- To use ball tracking model, it relies on TF Lite and selecting the correct delegate (NPU, GPU, CPU) and optimizing inference speed can be challenging
- Handling OpenCV native loading and TF Lite delegate management can cause compatibility issues across different Android versions and devices
- The label tool is using Matlab and it has some bugs on my laptop, trying to fix those bugs
- Some of the codes of the label tool seems to be fixed just for car park slots

### Xi Ding
## What have I done?
- Evaluate the performance of TrackNet V4

## What am I doing next?
- Explore Android app camera feed integration
- Explore methods of tennis court detection

### Yichi Zhang

## What have I done?

- Fix bugs
  - Format mismatch when loading the model, such as conflicts between SavedModel and H5 formats
  - The teacher model uses NCHW format while the student model uses NHWC format, which may cause errors during `ChannelConverter` transformation.
  - The teacher model accepts 9-channel input (three frames of images), while the student model accepts 3-channel input (a single frame)
  - Insufficient GPU memory, especially when the batch size is set too large.

## What am I doing next?

- Try to use the trained model to predict