import os
import time
import argparse
from functools import partial
import cv2
import torch
import openvino
import openvino.runtime as ov
import numpy as np

from utils.detector_utils import DataStreamer, save_output, non_max_suppression, preprocess_image

def load_model(model_xml_path: str, model_bin_path: str, target_device: str = "CPU"):
    # load IECore object
    OVIE = ov.Core()

    # # load CPU extensions if availabel
    # lib_ext_path = '/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.so'
    # if 'CPU' in target_device and os.path.exists(lib_ext_path):
    #     print(f"Loading CPU extensions from {lib_ext_path}")
    #     OVIE.add_extension(lib_ext_path, "CPU")

    # load openVINO network
    OVNet = OVIE.read_model(
        model=model_xml_path, weights=model_bin_path)

    # create executable network
    OVExec = OVIE.compile_model(
        OVNet, "CPU")
    print("Available Devices: ", OVIE.available_devices)

    return OVExec


def inference(input_path: str, OVExec, output_dir: str, threshold: float, save: bool = False, debug: bool = False) -> None:
    """Run Object Detection Application
    """
    if debug:
        print("Running Inference for {}: {}".format("image",input_path))

    # Get Input, Output Information
    input_blob = next(iter(OVExec.inputs))
    OutputLayer = list(OVExec.outputs)[-1]
    if debug:
        print("Input Layer: ", input_blob)
        print("Output Layer: ", OutputLayer)
        print("Model Input Shape: ",
            input_blob.shape)
        print("Model Output Shape: ", OutputLayer.shape)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    _, C, H, W = input_blob.shape
    preprocess_func = partial(preprocess_image, in_size=(W, H))
    data_stream = DataStreamer(input_path, "image", preprocess_func)
    OVExec.batch_size = 1
    for i, (orig_input, model_input) in enumerate(data_stream, start=1):
        if debug:
            start = time.time()

        # Inference
        infer_request = OVExec.create_infer_request()
        results = infer_request.infer(inputs={input_blob: model_input})
        output_tensor = infer_request.get_output_tensor()
        output_data = torch.from_numpy(output_tensor.data)    
        
        if debug: 
            end = time.time()
            inf_time = end - start
            print('Inference Time: {} Seconds Single Image'.format(inf_time))
            fps = 1. / (end - start)
            print('Estimated Inference FPS: {} FPS Single Image'.format(fps))

        detections = non_max_suppression(
            output_data, conf_thres=0.4, iou_thres=0.5)
        
        if save:
            save_path = os.path.join(
                output_dir, f"frame_openvino_{str(i).zfill(5)}.jpg")
            save_output(detections[0], orig_input, save_path,
                        threshold=threshold, model_in_HW=(H, W),
                        line_thickness=None, text_bg_alpha=0.0)


    if debug:    
        print(detections[0].shape)
        elapse_time = time.time() - start_time
        print(f'Total Frames: {i}')
        print(f'Total Elapsed Time: {elapse_time:.3f} Seconds'.format())
        print(f'Final Estimated FPS: {i / (elapse_time):.2f}')

base_path = "C:/Users/arite/Desktop/KMBIC/Collins Aerospace final codes/499-people-v1/"

OVExec = load_model(
    model_xml_path= base_path + "best_openvino_model/best.xml", 
    model_bin_path= base_path + "best_openvino_model/best.bin", 
    target_device= "CPU",)

inference(
    input_path= base_path + "valid/images/image1_jpg.rf.0a61c1987729ab9e39926aeb6468fade.jpg",
    OVExec= OVExec,
    output_dir= base_path + "best_openvino_model/output", 
    threshold= 0.6,
    debug= True)