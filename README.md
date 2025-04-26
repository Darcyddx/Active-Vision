The demo is revised from on [active_vision_qualcomm](https://gitee.com/michaellzy/active-vision-app) so there might be slight differences between this one and the version on main branch.

The pre-trained RNN model to recognize pose is originally from [tennis_shot_recognition](https://github.com/antoinekeller/tennis_shot_recognition). As it use TensorList, the H5 model cannot be convert to tflite directly, so there are two options, `Select TF Ops Delegate` (need extra delegate support on Android) or `pure TFLITE_BUILTINS` (big loss) , the runnable version now use the latter one. 

