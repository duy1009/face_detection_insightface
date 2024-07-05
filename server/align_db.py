import cv2, time, os
import numpy as np
from scrfd import SCRFD_INFERENCE, Face
# app = SCRFD_INFERENCE(model_path="src/weights/det_10g.onnx", root = "", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
from _process import count_angle, alignface
import glob
SAVE_DIR = "db_align" 
MODEL_PATH = "server/weights/det_10g.onnx"
SRC = "/home/hungdv/Downloads/Telegram Desktop/Anh_cham_cong_phong_IT_TC"
INPUT_SIZE = (640, 640)

def filter(pred, conf = 0.5, angle=30):
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

app = SCRFD_INFERENCE(model_path=MODEL_PATH)
app.prepare(ctx_id=0, input_size=INPUT_SIZE)

try: 
    os.makedirs(SAVE_DIR)
except:
    pass

pre = time.time()
fcnt= 0
fps = 0
cnt = 0
save_dis = False
dis = 0
paths = glob.glob(os.path.join(SRC, "**.jpg")) + glob.glob(os.path.join(SRC, "**.jpg"))
for path in paths:
    rimg = cv2.imread(path)

    pred = app.detect(rimg)
    faces_r = postprocess(pred)
    faces = filter(faces_r)
    img_show = rimg.copy()
    if len(faces)==1:
        json_pred = convertPred2Json(faces)
        bbox = json_pred["bbox"]
        imgs_f, imgs_size = alignCrop(rimg, faces)
        name_f = os.path.basename(path)
        cv2.imwrite(os.path.join(SAVE_DIR, name_f), imgs_f[0])
    else:
        print(f"[Number of faces != 1]:{path}")

    fcnt+=1
    cv2.putText(img_show, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (170, 0, 0), 1)
    cv2.imshow("Faces detection", img_show)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    # break

cv2.destroyAllWindows()