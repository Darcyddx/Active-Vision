### TrackNetV2 Finetune On Tennis Dataset

Load pre-trained weight (3-in-3-out), and transfer-learning on tennis dataset to track tennis trajectory.

#### Set up

1. Download pre-trained model [here](https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNetv2/-/blob/master/3_in_3_out/model906_30?ref_type=heads) and dataset

2. Generate pre-processing data and ground truth as `npz` files, need to modify `root_path` and `dataDir`
    ```
    python3 gen_input_data.py
    ```

3. Start training by executing:
    ```
    python3 train_finetune.py --load_weights=<path to model906_30> --save_weights=<path to save finetune model> --dataDir=<npzDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>
    ```

4. use `model_conversion_tflite.py` to convert saved model weights to tflite model