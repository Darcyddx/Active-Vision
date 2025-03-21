### TrackNetV2 Distillation

Use trained TrackNetV2 as teacher model and use a small ResNet+Unet model as student model to perform knowledge distillation on tennis dataset.

#### Required Package

```
tensorflow[and-cuda]==2.12.0
pandas
pillow
scikit-learn
matplotlib
opencv-python
[optional] tflite-runtime
```

#### Set up

1. Download pre-trained model(teacher model) [here](https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNetv2/-/blob/master/3_in_3_out/model906_30?ref_type=heads) and [dataset](https://anu365-my.sharepoint.com/:u:/g/personal/u7690985_anu_edu_au/ERHmo2uKHdxBkNqfoFpRa1kBf8ligCtsUvAmtlHYIYKogQ?e=c4criq)

2. Generate pre-processing data and ground truth as `npz` files, need to modify `root_path` and `dataDir`
    ```
    python3 gen_input_data.py
    ```

3. Start training by executing:
    ```
    python3 train_distillation.py --load_weights=<path to model906_30> --save_weights=<path to save finetune model> --dataDir=<npzDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>
    ```

4. use `model_conversion_tflite.py` to convert saved model weights to tflite model



#### Trained Weights

| SavedModel                                                   | TFLite                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Download]([tracknet_distillation_best_0320](https://anu365-my.sharepoint.com/:f:/g/personal/u7690985_anu_edu_au/Eg6ZqbJ4sV9HiyXIG0bm2WgBns2zPKQA4uGsid8OpJbptg?e=Z8UqcH)) | [Download]([tracknetv2_resnet_distillation_uint8.tflite](https://anu365-my.sharepoint.com/:u:/g/personal/u7690985_anu_edu_au/EeglyYm169lKt9RaQRmtD_QBdzn4M8K6YHXnZrgUFra1hg?e=vOfI8W)) |

