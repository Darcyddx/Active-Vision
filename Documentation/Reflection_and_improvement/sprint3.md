## Reflection on Team Communication and Problem Solving

### Specific event

During this sprint, I encountered a significant technical challenge while attempting to deploy our Detectron2 model to TensorFlow Lite (TFLite). The intended workflow involved converting the model first to ONNX, then to TensorFlow, and finally to TFLite. However, due to the unique architecture of Detectron2, I was only able to export it successfully to ONNX opset 16—and that too after making some custom modifications to the Detectron2 export scripts.

The real obstacle began when attempting to convert the ONNX model to TFLite. The onnx2tf tool failed due to unimplemented operations, and the onnx-tf library only supports opset versions up to 12, making it incompatible with our exported model. Despite trying several other conversion tools and approaches, every attempt to reach TFLite deployment failed.

At that point, I turned to my team and raised the issue in our communication channel. By discussing the problem openly, we collectively evaluated alternatives and made a pragmatic decision to keep the model in ONNX format and explore deploying it directly to the mobile application. Surprisingly, the ONNX model worked well in real-time performance tests on the app, ultimately fulfilling our core requirement.

This experience was an important reminder that open communication and team collaboration are essential, especially when facing complex technical roadblocks. Initially, I had been trying to solve the problem on my own, but reaching out allowed us to spot overlooked options and arrive at a viable solution more efficiently. It highlighted the value of diverse perspectives in problem-solving—something that’s easy to miss when working in isolation.

### Lessons Learned

- Don’t hesitate to ask for help: Discussing technical issues with the team early can uncover alternative solutions and save time.
- Be flexible with goals: While TFLite deployment was ideal, ONNX served as an effective fallback with acceptable performance.
- Communication channels are strategic tools: They’re not just for updates, but for brainstorming and decision-making under uncertainty.

### Actionable Improvements for the Future

- Set up short technical syncs or debugging huddles when someone is blocked, to accelerate resolution.
- Encourage early reporting of blockers in daily or weekly check-ins.
- Maintain a shared document of known limitations and alternatives for model deployment strategies to reduce trial-and-error in future projects.

## Reflection on successes

## Key lessons learned during the sprint 

## Actionable improvements

## Task estimations and velocity tracking

## Evaluation based on reflection