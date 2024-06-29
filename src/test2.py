import cv2, time
import numpy as np
from math import atan, pi
from scrfd import SCRFD_INFERENCE, Face

# app = SCRFD_INFERENCE(model_path="src/weights/det_10g.onnx", root = "", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app = SCRFD_INFERENCE(model_path="src/weights/det_10g.onnx")

app.prepare(ctx_id=0, input_size=(640, 640))
vid = cv2.VideoCapture("src/IMG_5272.MOV")

pre = time.time()
fcnt=0
FPS = 0
def filter(pred, conf = 0.5, angle=20):
    res = []
    for i in pred:
        ang = count_angle(i["kps"])
        print(ang)
        if ang < angle and i["det_score"]> conf :
            res.append(i)
    return res

def count_angle(landmark):
    a = landmark[2][1] - (landmark[0][1] + landmark[1][1]) / 2
    b = landmark[2][0] - (landmark[0][0] + landmark[1][0]) / 2
    angle = atan(abs(b) / a) * 180.0 / pi
    return angle
def cropFace(image_raw, bbox):
    x1, y1, x2, y2 = bbox.astype("int32")
    img_f = image_raw[y1:y2, x1:x2]
    return img_f

def cropFaces(image_raw, pred):
    face_imgs = []
    for res in pred:
        bbox = res["bbox"]
        face_imgs.append(cropFace(image_raw, bbox))
    return face_imgs
def postprocess(pred):
    bboxes, kpss = pred
    if bboxes.shape[0] == 0:
        return []
    ret = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        ret.append(face)
    return ret
            

while True:
    now = time.time()
    if now-pre>1:
        pre +=1
        FPS = fcnt
        fcnt=0
    r, img = vid.read()
    if not r:
        break
    pred = app.detect(img)
    faces = postprocess(pred)
    faces = filter(faces)
    if faces is not None:
        imgs_f = cropFaces(img, faces)
        rimg = app.draw_on(img, faces)
    fcnt+=1
    cv2.putText(rimg, f"FPS: {FPS}", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (170, 0, 0), 1)
    cv2.imshow("Faces detection", rimg)
    if cv2.waitKey(1) == ord("q"):
        break
    # break

cv2.destroyAllWindows()