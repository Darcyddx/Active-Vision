# ResNet_Track Distilled TrackNetV2

**To ensure the model can run smoothly on mobile devices, we need a model significantly smaller than TrackNetV2. The current model, which is reconstructed based on ResNet_Track and fine-tuned from TrackNetV2, exhibits noticeable issues such as ball loss and frame drops during actual detection. Based on the client's suggestion, we attempt to distill the pre-trained TrackNetV2 model using the ResNet_Track architecture to improve the model's generalization performance.**

Before run the code, install dependencies (change the version to adapt CUDA).

```bash
pip install -r requirements.txt
```



This `model_distillation.py` code is commonly used in the COCO dataset:

```bash
Dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
├── valid/
│   ├── image3.jpg
│   ├── image4.jpg
│   └── _annotations.coco.json
└── test/
    ├── image5.jpg
    ├── image6.jpg
    └── _annotations.coco.json
```



## Dataset References

| Name           | Description                                                  | URL                                                          |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| tennis-tracker | The "tennis-tracker" model could be used for detailed game analysis in broadcasting | [tennis-tracker Computer Vision Project](https://universe.roboflow.com/tennistracker-dogbm/tennis-tracker-duufq) |
| Tennis Model   | /                                                            | [Tennis Model Computer Vision Project](https://universe.roboflow.com/tennis-ai/tennis-model) |



### TrackNetV2 Model Architecture

![TrackNetV2 Model Architecture](C:\Users\Striker\OneDrive - Australian National University\桌面\ANU\25S1\COMP8715 Project\Active-Vision\resnet-track-architecture.svg)

### ResNet_Track Model Architecture

![ResNet_Track Model Architecture](C:\Users\Striker\OneDrive - Australian National University\桌面\ANU\25S1\COMP8715 Project\Active-Vision\resnet-track-architecture.svg)