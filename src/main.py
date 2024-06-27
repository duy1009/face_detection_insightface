from fastapi import FastAPI, UploadFile, File
from face_analysis_override import FaceAnalysisOverride as FaceAnalysis
from PIL import Image
import numpy as np
import os

app = FastAPI() 
ROOT = os.path.dirname(__file__)
fa = FaceAnalysis(model_path=os.path.join(ROOT, "weights", "det_10g.onnx"), 
                  root = "", 
                  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
fa.prepare(ctx_id=0, det_size=(640, 640))

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
        

@app.post("/extract-face") 
async def root(image: UploadFile = File(...)): 
    # contents = image.file.read()
    if image is None:
        return{"message": "Not found image"}
    image = Image.open(image.file).convert("RGB")
    img = np.array(image)[:, :, ::-1]
    faces = fa.get(img)
    faces = convertPred2Json(faces)
    if faces is not None:
        faces["state"] = 1
        return faces
    else:
        return {"state": 0}



