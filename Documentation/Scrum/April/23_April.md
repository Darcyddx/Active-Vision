# Daily - stand ups 23/04/25

### Pei Ling Lam
## What have I done?

- Train court dataset using Detectron2 model halfway through

## What am I doing next? 

- debugging on the training script

### Yichi Zhang

## What have I done?

- Select and fine-tune a suitable pose estimation model, learn RNN and LSTM, two basic methods to distinguish one shot pose

## What am I doing next? 

- Build RNN training code

### Xi Ding
## What have I done?
- Researched and identified key posture points relevant to the app’s use case and defined their visual representation 

## What am I doing next?
- Analyze the app’s UI layout to determine optimal placement for posture point overlays without obstructing critical elements

### Zhiyuan Lu

## What have I done?

- Implement the code in post-processing - for each 17 heatmap outputs, find the largest pixel value in heatmap and find that coordinate in heatmap.

## What am I doing next? 

- Implement subpixel refinement

### Tao Lu
## What have I done?
- The key points were smoothed by the univariate filter, and the jitter was significantly reduced after the test.

## What am I doing next? 
- No framework structure (skeleton_pairs=None) was used in the experiment; pure coordinate regression was preferred.