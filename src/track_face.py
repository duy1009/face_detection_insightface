import cv2, time
import numpy as np
from math import atan, pi
from scrfd import SCRFD_INFERENCE, Face
from tracker import CentroidTracker
# app = SCRFD_INFERENCE(model_path="src/weights/det_10g.onnx", root = "", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app = SCRFD_INFERENCE(model_path="src/weights/det_10g.onnx")

app.prepare(ctx_id=0, input_size=(640, 640))
vid = cv2.VideoCapture("src/IMG_5272.MOV")
track = CentroidTracker(50)
pre = time.time()
fcnt=0
FPS = 0
def filter(pred, conf = 0.5, angle=180):
    res = []
    for i in pred:
        ang = count_angle(i["kps"])
        # print(ang)
        if ang < angle and i["det_score"]> conf :
            res.append(i)
    return res
def convertPred2Json(pred):
    if len(pred)<=0:
        return None
    bboxes = np.empty((len(pred), 4))
    confs = np.empty((len(pred),))
    landmarks = np.empty((len(pred),5, 2))
    for i, pr in enumerate(pred):
        bboxes[i] = pr["bbox"]
        confs[i] = pr["det_score"]
        landmarks[i] = pr["kps"]
    
    return {
        "bbox": bboxes.tolist(),
        "det_score": confs.tolist(),
        "kps": landmarks.tolist()
    }
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
            
def findbbox(centers, bboxes):
    res = np.zeros((len(centers), 4), dtype="int")
    bboxes = np.array(bboxes)
    Centroids = np.zeros((len(bboxes), 2), dtype="int")
    Centroids[:,0] = ((bboxes[:,0] + bboxes[:,2]) / 2.0).astype("int")
    Centroids[:,1] = ((bboxes[:,1] + bboxes[:,3]) / 2.0).astype("int")

    for i, ct in enumerate(centers):
        t = None
        for ind, j in enumerate(Centroids):
            if j[0] == ct[0] and j[1] == ct[1]:
                t = bboxes[ind]
                break
        if t is not None:
            res[i] = t
    return res

def draw(img, id, bb):
    dimg = img.copy()
    for i, box in enumerate(bb):
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), (0, 150, 255), 2)
        cv2.putText(dimg, f"ID:{id[i]}", (box[0], max(0, box[1]-5)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 150, 255), 1)
    return dimg    

while True:
    now = time.time()
    if now-pre>1:
        pre +=1
        FPS = fcnt
        fcnt=0
    r, rimg = vid.read()
    if not r:
        break
    pred = app.detect(rimg)
    faces = postprocess(pred)
    faces = filter(faces)
    if len(faces)>0:
        json_pred = convertPred2Json(faces)
        bbox = json_pred["bbox"]
        # print(bbox)
        res = track.update(bbox)
        bbres = findbbox(list(res.values()), bbox)
        imgs_f = cropFaces(rimg, faces)
        img_show = app.draw_on(rimg, faces)
        img_show = draw(img_show, list(res.keys()), bbres)
    fcnt+=1
    cv2.putText(img_show, f"FPS: {FPS}", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (170, 0, 0), 1)
    cv2.imshow("Faces detection", img_show)
    if cv2.waitKey(1) == ord("q"):
        break
    # break

cv2.destroyAllWindows()