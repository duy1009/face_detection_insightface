from _process import count_angle, alignface 
import numpy as np
from scrfd import Face
import cv2
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