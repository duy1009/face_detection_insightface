import cv2, time, os, datetime
import os.path as osp
import numpy as np
from scrfd import SCRFD_INFERENCE, Face
from tracker import CentroidTracker2
from _process import count_angle, alignface
from logger import LogCSV
import requests

# API
HOST = '192.168.2.233'
PORT = '8000'

# Path
SAVE_DIR = "log" 
MODEL_PATH = "weights/det_10g.onnx"
SRC = 0

# Config time
TIME_REQUEST_NSTRANGER= 10  
TIME_REQUEST_STRANGER = 5
RESET_LOG = True

# Config face detection
DET_INPUT_SIZE = (640, 640)
DET_CONF_FIL = 0.5
FACE_ANGLE_FIL = 180

# Config tracking
TRACK_MAX_HIDE_FRAME = 10
SCORE_RATE = 0.5

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

def alignCrop(image_raw, pred):
    imgs = []
    sizes = []
    for res in pred:
        lmk = res["kps"]
        size = res["bbox"][2]-res["bbox"][0], res["bbox"][3]-res["bbox"][1]
        img = alignface(image_raw, lmk)
        imgs.append(img)
        sizes.append(size)
    return imgs, sizes

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
            
def drawTrack(img, track_pred):
    dimg = img.copy()
    for track in track_pred:
        if track['disappeared'] > 0: 
            continue
        box = track["bbox"].astype("int")
        id = track["id"]
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), (0, 150, 255), 2)
        cv2.putText(dimg, f"ID:{id}", (box[0], max(0, box[1]-5)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 150, 255), 1)
    return dimg    

def drawFace(img, track_pred, IDs):
    dimg = img.copy()
    for track in track_pred:
        if track['disappeared'] > 0: 
            continue
        for _ids in IDs:
            if track["id"] == _ids[0]:
                box = track["bbox"].astype("int")
                color = (0, 150, 255) if _ids[1][:8]!= "stranger" else (255, 105, 0)
                cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(dimg, f"{_ids[1]}", (box[0], max(0, box[1]-5)), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    return dimg    

def requestRecognizeFace(image, temp_path):
    cv2.imwrite(temp_path, image)
    files = {"image": open(temp_path, 'rb')}
    resp = requests.post(f'http://{HOST}:{PORT}/recognize_face', files=files)
    
    if resp.status_code == 200:
        data_recv = resp.json()
        return data_recv
    else:
        return None

def getListNotStranger(IDs):
    ids = {}
    for id, name in IDs.items():
        if len(name) >8 and name[:8] != "stranger":
            ids[id] = name
    return ids

def getScoreTrack(img_size_avg, angle, r=0.5):
    r1 = img_size_avg/112
    if img_size_avg > 1:
        r1 = 1
    r2 = 1 - abs(angle)/90
    return r1*r + r2*(1-r)

def trackIsBetter(track_res, _ids, rate = 0.5):
    track_angle = count_angle(track_res["kps"])
    track_res_img_size = (track_res["bbox"][2] - track_res["bbox"][0] + track_res["bbox"][3] - track_res["bbox"][1])/2

    old_img_size = (_ids[4][0] + _ids[4][1])/2

    sc1 = getScoreTrack(track_res_img_size, track_angle, rate)
    sc2 = getScoreTrack(old_img_size, _ids[5], rate)
    return sc1 > sc2
    
app = SCRFD_INFERENCE(model_path=MODEL_PATH)
app.prepare(ctx_id=0, input_size=DET_INPUT_SIZE)
vid = cv2.VideoCapture(SRC)
Track = CentroidTracker2(TRACK_MAX_HIDE_FRAME)

temp_path = osp.join(SAVE_DIR, "temp", "face.jpg")
if not osp.exists(osp.join(SAVE_DIR, "temp")):
    os.makedirs(osp.join(SAVE_DIR, "temp"))
path_log = osp.join(SAVE_DIR, "log.csv")
if RESET_LOG and osp.exists(path_log):
    os.remove(path_log)

log = LogCSV(path = path_log,
             header = ["ID", "Similarity", "Time", "State"],
             mode = 'a'
             )

if not RESET_LOG:
    log.update_a(["-", 0, str(datetime.datetime.now()), "Reset device!"])
pre = time.time()
fcnt= 0
fps = 0
save_dis = False
IDs = []
pre_ids = []
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
    faces = filter(faces_r, DET_CONF_FIL, FACE_ANGLE_FIL)
    img_show = rimg.copy()
    res = Track.update(faces)
    
    # re-recognition
    ids = []
    for _inx, (_id,_name, _t, _img_align, _img_size, _angle, _block) in enumerate(IDs):
        time_request = TIME_REQUEST_NSTRANGER if len(_name) >8 and _name[:8] != "stranger" else TIME_REQUEST_STRANGER
        if  now - _t > time_request and not _block:
            rep = requestRecognizeFace(IDs[_inx][3], temp_path)
            print("[Request]: re-recognition")
            if rep["name"] == "stranger":
                rep["name"] = "stranger"+str(_id)
            if rep["name"] != _name:
                new_name = rep["name"]
                log.update_a(["[Change]", rep["score"], str(datetime.datetime.now()), f"{_name} -> {new_name}"])
            IDs[_inx] = [_id, rep["name"], now, None, None, None, True]
        ids.append(_id)

    # Person in  
    track_ids = [] 
    for track_res in res:
        for _inx, (_id,_name, _t, _img_align, _img_size, _angle, _block) in enumerate(IDs):
            # Update better image
            if track_res["id"] == _id:
                if _img_align is None or trackIsBetter(track_res, IDs[_inx], SCORE_RATE):
                    imgs_f, imgs_size = alignCrop(rimg, [track_res])
                    IDs[_inx][3] = imgs_f[0]
                    IDs[_inx][4] = imgs_size[0]
                    IDs[_inx][5] = count_angle(track_res["kps"])
                    IDs[_inx][6] = False
        # Check new person
        if track_res["id"] not in ids:
            imgs_f, imgs_size = alignCrop(rimg, [track_res])
            rep = requestRecognizeFace(imgs_f[0], temp_path)
            print("[Request]: new person")
            if rep["name"] == "stranger":
                rep["name"] = "stranger"+str(track_res["id"])
            log.update_a([rep["name"], rep["score"], str(datetime.datetime.now()), "In"])
            IDs.append([track_res["id"], rep["name"],  now, None, None, None, True])
        track_ids.append(track_res["id"])
    
    # Person out
    ids_now = [i[0] for i in IDs]
    for _id, _name in pre_ids:
        if _id not in ids_now:
            log.update_a([_name,"-", str(datetime.datetime.now()), "Out"])
    pre_ids = [(i[0], i[1]) for i in IDs]        

    ids_temp = []
    for _ids in IDs: 
        if _ids[0] in track_ids:
            ids_temp.append(_ids)
    IDs = ids_temp
    
    fcnt+=1
    img_show = drawFace(img_show, res, IDs)
    cv2.putText(img_show, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (170, 0, 0), 1)
    cv2.imshow("Faces detection", img_show)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    # break

cv2.destroyAllWindows()