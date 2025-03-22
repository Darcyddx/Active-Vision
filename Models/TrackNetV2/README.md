### Tennis Tracking Based on TrackNetV2

#### Include Files

- Label-Tool: Contains a python script to label own tennis dataset
- TrackNet-Finetune: Includes the training code to fine-tune the TrackNet model on a tennis dataset. Additionally, it provides the functionality to convert the trained model into a TFLite format for inference on edge devices. The fine-tuned model achieves F1-score of 0.955 on the test set.
- TrackNet-Modify: This project modifies the original VGG16+UNet architecture into a ResNet+UNet architecture. It includes the functionality to convert the trained model into TFLite format for deployment on mobile devices. The modified model achieves an F1-score of 0.93 on the test set and supports real-time inference on mobile devices. However, in many cases, the model has a high FP2 score (False Positive: when the ball is predicted but not present in the ground truth).

#### Trained Weights

| Model              | # of Parameters | Trained Model | TFLite-FP16 | TFLite-UINT8 |
| ------------------ | --------------- | ------------- | ----------- | ------------ |
| TrackNet-Finetune  | 11.34M          |     [Saved Weight](https://drive.google.com/file/d/1DG9xNfF0EM2prFis84o4FKB4c5SVDd3F/view?usp=drive_link)          |    [Download](https://drive.google.com/file/d/1I2qunxk1Vl5tla0Xxm8hpIb73EPbHZim/view?usp=drive_link)         |     [Download](https://drive.google.com/file/d/1OO1f4n_cZK3xvY_URk2ebf9J5Z4HLeRc/view?usp=drive_link)         |
| TrackNet-Modify | 2.1M            |       [Saved Model](https://drive.google.com/drive/folders/1EYuj3tNkkEUJVLqJzWMg7iwIYvsXvzk7?usp=drive_link)        |    [Download](https://drive.google.com/file/d/1dq0WDfUE2NlKic2pUhnfr8f2vEiHFwy7/view?usp=drive_link)         |      [Download](https://drive.google.com/file/d/12L9z5rc2XyykUhD5QYji3f5FQdh_TGsH/view?usp=drive_link)        |

