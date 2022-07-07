# overhead_person_detector
Detect people in the vicinity of a door using overhead fisheye CCTV footage.

The model used is YOLOv5, trained on the Top View Multi Person and Sunmi in-office dataset, trained with 50 epochs with default settings, only class detected is 'person'. However, they mark the people differently, with earlier marking the full person but Sunmi marking just the head. This issue was noticed later, and we switched to a custom dataset from the place of implementation.

The model is then converted to a OpenVino model for improved performance with an accompanying script for detection due to the use of Intel CPU, and passing a boolean True/False for detection of people to the MiR robot integration on a FlaskJS server.

After training 50 epochs, model achieves mAP 0.5:0.95 of 0.7 and mAP 0.5 of 0.97. Results are available at [Weights and Biases](https://wandb.ai/aribic/YOLOv5/runs/auasy8xw/overview).

Detection is mainly based on work from https://github.com/SamSamhuns/yolov5_export_cpu, but has been improved to use OpenVino Runtime API 2.0
Performance is ~3FPS on Intel i5-8265U.

To improve, YOLO-OBB can be used for oriented bounding boxes, however performance is still good on this version.

