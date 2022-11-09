import os
import openvino_tensorflow as ovtf
from yolov5_utils import  infer_openvino_tensorflow
os.environ["OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS"] = "1"

def main():
    input_file = "data/images/zidane.jpg"
    model_file = "yolov5s_saved_model" 
    label_file = "coco.names"
    input_height = 640
    input_width = 640
    backend_name = "CPU"
    conf_threshold = 0.6
    iou_threshold = 0.5

    #Print list of available backends
    print('Available Backends:')
    backends_list = ovtf.list_backends()
    for backend in backends_list:
        print(backend)
    ovtf.set_backend(backend_name)

    print("OpenVINO TensorFlow is enabled")
    infer_openvino_tensorflow(model_file, input_file, input_height, input_width, label_file, conf_threshold, iou_threshold)

    ovtf.disable() ## Disabling OVTF
    print("OpenVINO TensorFlow is disabled")
    infer_openvino_tensorflow(model_file, input_file, input_height, input_width, label_file, conf_threshold, iou_threshold)
    
if __name__=="__main__":
    main()
