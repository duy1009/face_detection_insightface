import cv2, time, os, datetime
import os.path as osp
import numpy as np
from scrfd import SCRFD_INFERENCE, Face
from tracker import CentroidTracker2
from _process import count_angle, alignface
from logger import LogCSV
import requests

HOST = '127.0.0.1'
PORT = '8000'

SAVE_DIR = "log" 
MODEL_PATH = "src/weights/det_10g.onnx"
SRC = "/home/hungdv/Downloads/video_test/Duy333333333.avi"
INPUT_SIZE = (640, 640)

TIME_REQUEST_NSTRANGER= 10  
TIME_REQUEST_STRANGER = 5

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
            
def draw_track(img, track_pred):
    dimg = img.copy()
    for track in track_pred:
        if track['disappeared'] > 0: 
            continue
        box = track["bbox"].astype("int")
        id = track["id"]
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), (0, 150, 255), 2)
        cv2.putText(dimg, f"ID:{id}", (box[0], max(0, box[1]-5)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 150, 255), 1)
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
    r2 = 1 - abs(angle)/180
    return r1*r + r2*(1-r)


def trackIsBetter(track_res, _ids, rate = 0.5):
    track_angle = count_angle(track_res["kps"])
    track_res_img_size = (track_res["bbox"][2] - track_res["bbox"][0] + track_res["bbox"][3] - track_res["bbox"][1])/2

    old_img_size = (_ids[4][0] + _ids[4][1])/2

    sc1 = getScoreTrack(track_res_img_size, track_angle)
    sc2 = getScoreTrack(old_img_size, _ids[5])
    return sc1 > sc2
    


app = SCRFD_INFERENCE(model_path=MODEL_PATH)
app.prepare(ctx_id=0, input_size=INPUT_SIZE)
vid = cv2.VideoCapture(SRC)
Track = CentroidTracker2(5)

temp_path = osp.join(SAVE_DIR, "temp", "face.jpg")
try: 
    os.makedirs(SAVE_DIR)
except:
    pass
try:
    os.makedirs(osp.join(SAVE_DIR, "temp"))
except:
    pass
log = LogCSV(osp.join(SAVE_DIR, "log.csv"),
             ["ID", "Time", "State"],
             "a"
             )

pre = time.time()
fcnt= 0
fps = 0
save_dis = False
IDs = []
while True:
    now = time.time()
    if now-pre>1:
        pre +=1
        fps = fcnt
        fcnt=0
    r, rimg = vid.read()
    # rimg = cv2.imread(path)
    if not r:
        break
    pred = app.detect(rimg)
    faces_r = postprocess(pred)
    faces = filter(faces_r)
    img_show = rimg.copy()
    res = []

    res = Track.update(faces)
    img_show = app.draw_on(rimg, faces)
    # print(res)
    img_show = draw_track(img_show, res)

    # Person in  
    track_ids = []   
    persons_in = []
    ids = []
    for _inx, (_id,_name, _t, _img_align, _img_size, _angle, _block) in enumerate(IDs):
        time_request = TIME_REQUEST_NSTRANGER if len(_name) >8 and _name[:8] != "stranger" else TIME_REQUEST_STRANGER
        if  now - _t > time_request and not _block:
            # imgs_f, imgs_size = alignCrop(rimg, [track_res])
            rep = requestRecognizeFace(IDs[_inx][3], temp_path)
            print("[Request]: re-recognition")
            if rep["name"] == "stranger":
                rep["name"] = "stranger"+str(_id)
            if rep["name"] != _name:
                new_name = rep["name"]
                log.update_a(["[Change]", str(datetime.datetime.now()), f"{_name} -> {new_name}"])
            IDs[_inx] = [_id, rep["name"], now, None, None, None, True]
        ids.append(_id)
    for track_res in res:
        for _inx, (_id,_name, _t, _img_align, _img_size, _angle, _block) in enumerate(IDs):
            if track_res["id"] == _id:
                imgs_f, imgs_size = alignCrop(rimg, [track_res])
                if _img_align is None or trackIsBetter(track_res, IDs[_inx]):
                    IDs[_inx][3] = imgs_f[0]
                    IDs[_inx][4] = imgs_size[0]
                    IDs[_inx][5] = count_angle(track_res["kps"])
                    IDs[_inx][6] = False
                
        if track_res["id"] not in ids:
            # persons_in.append(track_res)
            imgs_f, imgs_size = alignCrop(rimg, [track_res])
            rep = requestRecognizeFace(imgs_f[0], temp_path)
            print("[Request]: new person")
            if rep["name"] == "stranger":
                rep["name"] = "stranger"+str(track_res["id"])
            log.update_a([rep["name"], str(datetime.datetime.now()), "In"])
            IDs.append([track_res["id"], rep["name"],  now, None, None, None, True])
        track_ids.append(track_res["id"])
    
    # imgs_f, imgs_size = alignCrop(rimg, persons_in)
    # for img_f, ps in zip(imgs_f, persons_in):
    #     rep = requestRecognizeFace(img_f, temp_path)
    #     print(rep)


    # Person out
    ids_temp = []
    for _ids in IDs: # check doan nay ***********************************************************
        if _ids[0] in track_ids:
            ids_temp.append(_ids)
        if _ids[0] not in ids:
            log.update_a([_ids[1], str(datetime.datetime.now()), "Out"])
            
    IDs = ids_temp


    fcnt+=1
    cv2.putText(img_show, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (170, 0, 0), 1)
    cv2.imshow("Faces detection", img_show)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    # break

cv2.destroyAllWindows()