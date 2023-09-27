<div align="center">
  <img width="450" src="assets/Flask_logo.svg">
</div>

# Yolov5 Flask API for detection and segmentation

<div align="center">
  <a href="https://www.buymeacoffee.com/hdnh2006" target="_blank">
    <img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee">
  </a>
</div>

![Screen GIF](assets/screen.gif)

This code is based on the YOLOv5 from Ultralytics and it has all the functionalities that the original code has:
- Different source: images, videos, webcam, RTSP cameras.
- All the weights are supported: TensorRT, Onnx, DNN, openvino.

The API can be called in an interactive way, and also as a single API called from terminal. 



## Requirements

Python 3.8 or later with all [requirements.txt](requirements.txt) dependencies installed, including `torch>=1.7`. To install run:

```bash
$ pip3 install -r requirements.txt
```

## Object detection API

`detect_api.py` can deal with several sources and can run into the cpu, but it is highly recommendable to run in gpu.

```bash
Usage - sources:
    $ python detect_api.py --weights yolov5s.pt --source 0                               # webcam
                                                         img.jpg                         # image
                                                         vid.mp4                         # video
                                                         screen                          # screenshot
                                                         path/                           # directory
                                                         list.txt                        # list of images
                                                         list.streams                    # list of streams
                                                         'path/*.jpg'                    # glob
                                                         'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                         'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python detect_api.py --weights yolov5s.pt                 # PyTorch
                                     yolov5s.torchscript        # TorchScript
                                     yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                     yolov5s_openvino_model     # OpenVINO
                                     yolov5s.engine             # TensorRT
                                     yolov5s.mlmodel            # CoreML (macOS-only)
                                     yolov5s_saved_model        # TensorFlow SavedModel
                                     yolov5s.pb                 # TensorFlow GraphDef
                                     yolov5s.tflite             # TensorFlow Lite
                                     yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                     yolov5s_paddle_model       # PaddlePaddle
```


## Instance segmentation API

`segment_api.py` can deal with several sources and can run into the cpu, but it is highly recommendable to run in gpu.

```bash
Usage - sources:
    $ python segment_api.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  list.txt                        # list of images
                                                                  list.streams                    # list of streams
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python segment_api.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
```

## Classify API

`classify_api.py` can deal with several sources and can run into the cpu, but it is highly recommendable to run in gpu.

```bash
Usage - sources:
    $ python classify_api.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  list.txt                        # list of images
                                                                  list.streams                    # list of streams
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python classify_api.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
```


## Interactive implementation implemntation

You can deploy the API able to label an interactive way.

Run:

```bash
$ python detect_api.py --device cpu # to run into cpu (by default is gpu)
```
Open the application in any browser 0.0.0.0:5000 and upload your image or video as is shown in video above.


## How to use the API

### Interactive way
Just open your favorite browser and go to 0.0.0.0:5000 and intuitevely load the image you want to label and press the buttom "Upload image".

The API will return the image or video labeled.

### Call from terminal or python program
The `client.py` code provides several example about how the API can be called. A very common way to do it is to call a public image from url and to get the coordinates of the bounding boxes:

```python
import requests

resp = requests.get("http://0.0.0.0:5000/detect?url=https://atlassafetysolutions.com/wp/wp-content/uploads/2019/06/ppe.jpeg&save_txt=T",
                    verify=False)
print(resp.content)

```
And you will get a json with the following data:

```
b'{"results": [{"class": 72, "x": 0.647187, "y": 0.495779, "w": 0.421875, "h": 0.991557, "conf": null}, {"class": 0, "x": 0.371563, "y": 0.497655, "w": 0.525625, "h": 0.982176, "conf": null}]}'
```


## About me and contact

This code is based on the YOLOv5 from Ultralytics and it has been modified by Henry Navarro
 
If you want to know more about me, please visit my blog: henrynavarro.org.
