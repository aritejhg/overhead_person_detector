import os
import time
from functools import partial
from xmlrpc.client import boolean
import torch
import openvino.runtime as ov
from zmq import device

from detector_utils import DataStreamer, save_output, non_max_suppression, preprocess_image

# COORDS for masking
COORDS = [[231,340],[491,320],[751,344],[938,571],[644,847],[298,856],[23,597],[115,450],[231,350]]

def load_model(model_xml_path: str, model_bin_path: str, target_device: str = "CPU"):
    """
    Returns an executable OpenVino model given the xml and weights bin file paths"""
    # load IECore object
    OVIE = ov.Core()

    # load openVINO network
    OVNet = OVIE.read_model(
        model=model_xml_path, weights=model_bin_path)

    # create executable network
    OVExec = OVIE.compile_model(
        OVNet, target_device)
    print("Available Devices: ", OVIE.available_devices)

    return OVExec


def inference(input_path: str, OVExec, output_dir: str, conf_threshold: float = 0.55, iou_threshold: float = 0.6, save: bool = False, debug: bool = False) -> boolean:
    """Run Object Detection Application on an OpenVino executable network. 
    Takes in the img path, executable model, output path, confidence threshold, iou threshold, save and debug options. 
    Returns a boolean to tell if people are present in the region.
    """
    if debug:
        print("Running Inference for {}: {}".format("image",input_path))

    # Get Input, Output Information
    input_blob = next(iter(OVExec.inputs))
    OutputLayer = list(OVExec.outputs)[-1]
    if debug:
        print("Model Input Shape: ",  input_blob.shape)
        print("Model Output Shape: ", OutputLayer.shape)

    # make output directory
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    #init datastream
    start_time = time.time()
    _, C, H, W = input_blob.shape

    #create the preprocessing function and DataStream
    preprocess_func = partial(preprocess_image, in_size=(W, H))
    data_stream = DataStreamer(input_path, "image", preprocess_func)
    OVExec.batch_size = 1

    #start inference
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

        # Takes the raw output torch.tensor from output_data, applies NMS to give the final detections in the form of [n,6] matrix where n is number of people.
        # for each detection, xyxy confidence and class are provided.
        detections = non_max_suppression(
            output_data, conf_thres=conf_threshold, iou_thres=iou_threshold)

        if debug:
            print (detections)
        
        # saves output onto the original image if necessary        
        if save:
            save_path = os.path.join(
                output_dir, f"{input_path.split('/')[-1]}") 
            save_output(detections[0], orig_input, save_path,
                        threshold=conf_threshold, model_in_HW=(H, W),
                        line_thickness=None, text_bg_alpha=0.0)


    if debug:    
        print(f'{detections[0].shape[0]} people are detected')
        elapse_time = time.time() - start_time
        print(f'Total Frames: {i}')
        print(f'Total Elapsed Time: {elapse_time:.3f} Seconds'.format())
        print(f'Final Estimated FPS: {i / (elapse_time):.2f}')

    #returns if people are detected, as detections is a [n,6] matrix where n is number of people detected
    return detections[0].shape[0]>0


# testing 

# base_path = "C:/Users/arite/Desktop/KMBIC/Collins Aerospace final codes/499-people-v1/"

# OVExec = load_model(
#     model_xml_path= base_path + "overhead_person_detector/vanilla_150ep_openvino/best.xml", 
#     model_bin_path= base_path + "overhead_person_detector/vanilla_150ep_openvino/best.bin", 
#     target_device= "CPU")
# import fnmatch
# # for input in fnmatch.filter(os.listdir("C:/Users/arite/Desktop/KMBIC/Collins Aerospace final codes/499-people-v1/mask_test"), '*.jpg'):
# inference(
#         input_path= "C:/Users/arite/Desktop/KMBIC/datasets/test_set_collins/og/image(normal)1.jpg",
#         OVExec= OVExec,
#         output_dir= "C:/Users/arite/Desktop/KMBIC/Collins Aerospace final codes/499-people-v1/", 
#         threshold= 0.55,
#         debug= True, save = True)

# IMPLEMENT SAVE_PREV
# implement timestamp 
# documentation