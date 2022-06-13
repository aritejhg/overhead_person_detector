# overhead_person_detector
Detect people in the vicinity of a door using overhead CCTV footage.

The model used is YOLOv5, trained on the Top View Multi Person and Sunmi in-office dataset, trained with 50 epochs with default settings, only class detected is 'person'.
The model is then converted to a tflite model for improved performance with an accompanying script for detection, and passing a boolean True/False for detection of people.

After training 50 epochs, model achieves mAP 0.5:0.95 of 0.7 and mAP 0.5 of 0.97. Results are available at [Weights and Biases](https://wandb.ai/aribic/YOLOv5/runs/auasy8xw/overview).
