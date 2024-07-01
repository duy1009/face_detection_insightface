import cv2, time, os
import numpy as np
from math import atan, pi
from scrfd import SCRFD_INFERENCE, Face
from tracker import CentroidTracker
# app = SCRFD_INFERENCE(model_path="src/weights/det_10g.onnx", root = "", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
from _process import count_angle, alignface
from logger import LogCSV

SAVE_DIR = "result" 

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

def cropFace(image_raw, bbox):
    bbox = bbox.astype("int32")
    bbox[bbox<0] = 0
    x1, y1, x2, y2 = bbox
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




app = SCRFD_INFERENCE(model_path="src/weights/det_10g.onnx")

app.prepare(ctx_id=0, input_size=(640, 640))
vid = cv2.VideoCapture("src/IMG_5272.MOV")
track = CentroidTracker(50)

logg = LogCSV(os.path.join(SAVE_DIR, "result.csv"), 
               ["Size_w", "Size_h", "Confident", "Time_e2e(s)", "Angle(degree)", "Path"])

try: 
    os.makedirs(SAVE_DIR)
    os.makedirs(os.path.join(SAVE_DIR, "images"))
except:
    pass

pre = time.time()
fcnt= 0
fps = 0
cnt = 0
while True:
    now = time.time()
    if now-pre>1:
        pre +=1
        fps = fcnt
        fcnt=0
    r, rimg = vid.read()
    if not r:
        break
    pred = app.detect(rimg)
    faces_r = postprocess(pred)
    faces = filter(faces_r)
    if len(faces)>0:
        json_pred = convertPred2Json(faces)
        bbox = json_pred["bbox"]
        # print(bbox)
        res = track.update(bbox)
        bbres = findbbox(list(res.values()), bbox)
        imgs_f = cropFaces(rimg, faces)
        for img_f, face in zip(imgs_f, faces):
            lmk = face["kps"].copy()
            lmk[:,0]-= face["bbox"][0]
            lmk[:,1]-= face["bbox"][1]
            img_aligned = alignface(img_f, lmk)

            save_path = os.path.join(SAVE_DIR, "images", f"face_{cnt}.png")
            cv2.imwrite(save_path, img_aligned)
            angle = count_angle(lmk)
            proc_time = time.time()-now
            logg.update([img_f.shape[1],
                         img_f.shape[0],
                         face["det_score"],
                         proc_time,
                         angle,
                         save_path
                         ])
            cnt+=1
            
        img_show = app.draw_on(rimg, faces)
        img_show = draw(img_show, list(res.keys()), bbres)
    logg.save()
    fcnt+=1
    cv2.putText(img_show, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (170, 0, 0), 1)
    cv2.imshow("Faces detection", img_show)
    if cv2.waitKey(1) == ord("q"):
        break
    # break

cv2.destroyAllWindows()