from fastapi import FastAPI, UploadFile, File
from arcface import ArcFaceONNX
from PIL import Image
import glob, os
import numpy as np

app = FastAPI() 
MODEL_PATH = "/home/hungdv/tcgroup/Jetson/insightface/arc_R50.onnx"
DB = "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/db"
handler = ArcFaceONNX(MODEL_PATH)
handler.prepare(ctx_id=0)


# Load faces feature from database
db_img_path = glob.glob(os.path.join(DB, "**.png")) + glob.glob(os.path.join(DB, "**.jpg"))
feat_db = {}
for img_path in db_img_path:
    pil_image = Image.open(img_path).convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    im = open_cv_image[:, :, ::-1].copy()
    name = os.path.basename(img_path)[:-4]
    feat_db[name] = handler.get2(im)

def getSimilarFace(img):
    open_cv_image = np.array(img)
    # Convert RGB to BGR
    img = open_cv_image[:, :, ::-1].copy()
    feat2 = handler.get2(img)
    score_max = -2
    feat_same = []
    for feat_name, feat in feat_db.items():
        score = handler.compute_sim(feat, feat2)
        if score > score_max:
            feat_same = [feat_name, score]
            score_max = score
    if feat_same[1] < 0.3:
        feat_same[0] = "stranger"
    return {
        "name": feat_same[0],
        "score": float(feat_same[1])
    }

@app.post("/recognize_face") 
async def root(image: UploadFile = File(...)): 
    # contents = image.file.read()
    if image is None:
        return{"message": "Not found image"}
    image = Image.open(image.file).convert("RGB")

    return getSimilarFace(image)
