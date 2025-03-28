# Meeting Details

- **Date:** 18 Mar 2025
- **Time:** 2 PM – 3.40 PM
- **Location:** Zoom
- **Attendees:** All Team Members
- **Recorder:** Pei Ling Lam

## Agenda Items

1. **Discussion on Player detection task allocation**
   - Zhiyuan Lu has set up the Android app in GitHub repo
   - move ball tracking algorithm to Android app - Pei Ling Lam
   - player detector model - Zhiyuan Lu
   - player pose tracker - Xingchen Zhang
   - other functions:
     - tflite helpers - Zhiyuan Lu
     - codes for multiple threadings - Zhiyuan Lu
     - camera fragment - Tao Lu
     - codes to analyze the frames - Yichi Zhang
     - 20% of unexpected functions needed - Xi Ding

## Risk Management

### Risk Management for Ball Tracking in Mobile Deployment

In this project, ball tracking is a critical component that directly impacts the overall functionality and user experience of the application. However, deploying an effective ball tracking model on mobile devices presents significant challenges due to hardware limitations and performance constraints. Existing ball tracking models are typically large and computationally intensive, making them unsuitable for direct deployment on mobile platforms. Additionally, issues such as frequent ball loss, high noise levels, and inconsistent tracking accuracy further complicate the implementation. To mitigate these risks and ensure the success of the project, we have developed a comprehensive risk management strategy. Below, we outline the identified risks, their potential impact, and the proposed mitigation strategies.

---

#### Identified Risks and Their Implications

1. **Model Size and Computational Complexity**
   - **Risk Description**: Current ball tracking models, such as TrackNetV2, are designed for desktop environments with powerful GPUs. These models are too large and computationally demanding to run efficiently on mobile devices, which have limited processing power and memory.
   - **Impact**: Direct deployment of such models on mobile devices would result in slow inference speeds, high battery consumption, and poor user experience. This could lead to the failure of the application in meeting its functional requirements.

2. **Tracking Accuracy and Noise Handling**
   - **Risk Description**: Ball tracking models often struggle with maintaining consistent accuracy, especially in scenarios with occlusions, fast motion, or complex backgrounds. Additionally, noise from lighting conditions, shadows, or other visual artifacts can degrade performance.
   - **Impact**: Frequent ball loss and inaccurate tracking would undermine the reliability of the application, leading to user frustration and dissatisfaction. This could also affect downstream tasks that depend on accurate ball tracking, such as real-time analytics or augmented reality features.

3. **Architecture Limitations**
   - **Risk Description**: The use of outdated or inefficient architectures (e.g., VGG) may limit the model's ability to achieve optimal performance within the constraints of mobile hardware.
   - **Impact**: Suboptimal architecture choices could result in poor trade-offs between accuracy and computational efficiency, making it difficult to achieve the desired balance for mobile deployment.

4. **Knowledge Transfer and Model Adaptation**
   - **Risk Description**: Transitioning from existing models to more efficient ones without losing performance requires careful knowledge transfer and fine-tuning. Poor adaptation could lead to degraded performance or failed deployments.
   - **Impact**: If knowledge distillation or fine-tuning processes are not executed effectively, the resulting model may fail to meet accuracy requirements, rendering the entire effort ineffective.

---

#### Proposed Mitigation Strategies

To address the identified risks, we propose the following strategies:

1. **Model Pruning and Quantization**
   - **Description**: Model pruning involves removing redundant weights and neurons from the neural network to reduce its size and complexity. Quantization, on the other hand, reduces the precision of the model's parameters (e.g., from 32-bit floating-point to 8-bit integers), further decreasing computational demands.
   - **Implementation**:
     - Use tools like TensorFlow Lite or PyTorch Mobile to perform post-training quantization.
     - Apply structured pruning techniques to remove less important layers or neurons while preserving critical features.
   - **Expected Outcomes**: A significantly smaller and faster model that can run efficiently on mobile devices without a substantial drop in accuracy.

2. **Architecture Replacement: ResNet over VGG**
   - **Description**: Replace the current VGG-based architecture with a more modern and efficient ResNet architecture. ResNet's residual connections allow for deeper networks with better gradient flow, making it more suitable for complex tasks like ball tracking.
   - **Implementation**:
     - Fine-tune a pre-trained ResNet model on our ball tracking dataset.
     - Experiment with different ResNet variants (e.g., ResNet-18, ResNet-34) to find the best balance between performance and efficiency.
   - **Expected Outcomes**: Improved accuracy and reduced computational overhead compared to VGG-based models.

3. **Knowledge Distillation with Reconstructed Architecture**
   - **Description**: Knowledge distillation involves training a smaller "student" model to mimic the behavior of a larger "teacher" model. In this case, we will use a reconstructed architecture to distill knowledge from TrackNetV2 into a more compact model.
   - **Implementation**:
     - Design a lightweight student model tailored for mobile deployment.
     - Train the student model using soft labels generated by the teacher model (TrackNetV2).
     - Incorporate additional regularization techniques to prevent overfitting during distillation.
   - **Expected Outcomes**: A highly efficient model that retains most of the accuracy of the original TrackNetV2 while being suitable for mobile devices.

4. **Evaluation of TrackNetV4**
   - **Description**: Explore the feasibility of using TrackNetV4, a newer version of the TrackNet architecture, to see if it offers improvements in performance, efficiency, or robustness compared to TrackNetV2.
   - **Implementation**:
     - Benchmark TrackNetV4 against TrackNetV2 on our dataset to evaluate metrics such as accuracy, speed, and resource usage.
     - Assess its compatibility with mobile deployment frameworks like TensorFlow Lite or ONNX Runtime.
   - **Expected Outcomes**: If TrackNetV4 proves superior, it could serve as a drop-in replacement for TrackNetV2, providing better performance with minimal additional effort.

---

#### Additional Considerations

1. **Data Augmentation and Preprocessing**
   - To improve the robustness of the model, we will implement advanced data augmentation techniques, such as random cropping, rotation, and color jittering. Preprocessing steps like background subtraction and noise filtering will also be applied to enhance input quality.

2. **Continuous Monitoring and Feedback**
   - Post-deployment, we will establish a monitoring system to track model performance in real-world scenarios. User feedback will be collected to identify edge cases and areas for improvement.

3. **Fallback Mechanisms**
   - In cases where the primary tracking model fails (e.g., due to extreme occlusion or noise), we will implement fallback mechanisms such as heuristic-based tracking or manual correction options to maintain usability.

4. **Iterative Optimization**
   - The optimization process will be iterative, with regular evaluations and refinements based on performance metrics. This ensures that the final solution meets all project requirements.

---

#### Conclusion

The successful deployment of a ball tracking model on mobile devices hinges on addressing the challenges posed by hardware limitations, model complexity, and environmental factors. By implementing the proposed strategies—model pruning and quantization, architecture replacement, knowledge distillation, and evaluation of newer architectures—we aim to create a robust, efficient, and accurate solution. Additionally, continuous monitoring and iterative optimization will ensure long-term success and adaptability. Through this comprehensive risk management approach, we are confident in mitigating potential failures and delivering a high-quality product that meets user expectations. 

**Final Recommendation**: Begin with model pruning and quantization as the first step, followed by experimentation with ResNet and knowledge distillation. Simultaneously, evaluate TrackNetV4 as a potential alternative to streamline the process.