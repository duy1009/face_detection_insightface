### Introduction
This source code is face recognition for jetson Nano based insighface without use insightface python lib

### Support file:
1. arcface.py: contain face recoginition inference class from onnx model
2. _process.py: image processing (alignface,...)
3. logger.py: log csv class
4. scrfd.py: face detection inference class from onnx model
5. tracker.py: center bounding boxes tracking class

### Run file:
1. capture.py: capture video from camera -> .avi
2. eval_rec.py: evaluate recoginition model
Prepare: 
- Images (database)
- Folder crresponding to images in the database \
Result: log file similar, dissimilar -> draw histogram -> threshold
3. merge_report.py: Support merge detection and recoginition result.csv
4. recognition.py: face recognition using arcface
Input: folder images (aligned), database
Output: result.csv, folders image 
5. rename_detect_result.py: Support rename detect result (avoid duplication if there are multiple results need merge)
6. track_face.py: tracking face and save report detection (result.csv, face aligned images)
7. test2.py: test detection model (don't save any results)
8. main.py: detect face -> tracking -> using face recognition API -> log time "In" and time "Out"

### Export tensorrt on Jetson Nano
```/usr/src/tensorrt/bin/trtexec --onnx=resnet50/model.onnx --saveEngine=resnet_engine.trt```