# ResNet_Track Distilled TrackNetV2

**To ensure the model can run smoothly on mobile devices, we need a model significantly smaller than TrackNetV2. The current model, which is reconstructed based on ResNet_Track and fine-tuned from TrackNetV2, exhibits noticeable issues such as ball loss and frame drops during actual detection. Based on the client's suggestion, we attempt to distill the pre-trained TrackNetV2 model using the ResNet_Track architecture to improve the model's generalization performance.**



