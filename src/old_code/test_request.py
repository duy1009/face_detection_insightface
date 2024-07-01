import requests
import numpy as np
import cv2, time, os
from math import atan, pi
HOST = '0.0.0.0'
PORT = '8000'
CONFIDENT = 0.5
FACE_ANGLE = 20

image_path = r"/home/hungdv/tcgroup/Jetson/Face-Recognition-Jetson-Nano/img/Du.jpg"
def convertJson2Pred(json_data):
    if not json_data:
        return None

    bboxes = np.array(json_data["bbox"])
    confs = np.array(json_data["det_score"])
    landmarks = np.array(json_data["kps"])

    pred = []
    for i in range(len(bboxes)):
        pr = {
            "bbox": bboxes[i],
            "det_score": confs[i],
            "kps": landmarks[i]
        }
        pred.append(pr)
    return pred

def extractFace(img_path):
    files = {"image": open(img_path, 'rb')}
    resp = requests.post(f'http://{HOST}:{PORT}/extract-face', files=files)
    
    if resp.status_code == 200:
        data_recv = resp.json()
        return data_recv
    else:
        return None



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

def draw_on(img, faces):
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face["bbox"].astype(np.int32)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face["kps"] is not None:
                kps = face["kps"].astype(np.int32)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)

        return dimg

try:
    os.mkdir("./temp")
except:
    pass

while True:
    now = time.time()
    if now-pre>1:
        pre +=1
        FPS = fcnt
        fcnt=0
    r, rimg = vid.read()
    if not r:
        break
    cv2.imwrite("./temp/img.jpg", rimg)
    
    resp = extractFace("./temp/img.jpg")
    if resp["state"] == 1:
        faces = convertJson2Pred(resp)
        faces = filter(faces, CONFIDENT, FACE_ANGLE)
        if len(faces)>0:
            imgs_f = cropFaces(rimg, faces)
            rimg = draw_on(rimg, faces)
    fcnt+=1
    cv2.putText(rimg, f"FPS: {FPS}", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
    cv2.imshow("Faces detection", rimg)
    if cv2.waitKey(1) == ord("q"):
        break
    # break
try:
    os.remove("./temp/img.jpg")
except:
    pass
cv2.destroyAllWindows()
