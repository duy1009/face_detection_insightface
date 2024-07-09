from fastapi import FastAPI, UploadFile, File, Body
from arcface import ArcFaceONNX
from PIL import Image
import glob, os, datetime, time
import numpy as np
import pandas as pd

app = FastAPI() 
MODEL_PATH = "/home/hungdv/tcgroup/Jetson/insightface/arc_R50.onnx"
DB = "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/db"
LOG_DIR = "log"
TIME_IS_DISCONNECT_SYNC = 20

def convertStr2Time(str):
    return datetime.datetime.strptime(str, ' %Y-%m-%d %H:%M:%S.%f')

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

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
    feat_same[1] = (1+feat_same[1])/2
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

pre_time = time.time()
log_disconnect = False
@app.get("/last_time_log") 
def syncLastTime(): 
    global pre_time, log_disconnect
    now = time.time()
    if now - pre_time > TIME_IS_DISCONNECT_SYNC:
        log_disconnect = True
    pre_time = now
    # contents = image.file.read()
    files = glob.glob(os.path.join(LOG_DIR, "**.csv"))
    pre_date = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d").date()
    empty = True
    for file in files:
        try:
            f_date = datetime.datetime.strptime(os.path.basename(file)[4:-4], "%Y-%m-%d").date()
        except: continue
        if f_date>pre_date:
            pre_date = f_date
            empty = False
    if not empty:
        file_path = os.path.join(LOG_DIR, f"log_{str(pre_date)}.csv")
        df = pd.read_csv(file_path).values
        if len(df)>0:
            frame = df[-1]
            fr_date_time = frame[2]
        else:
            fr_date_time = f" {pre_date} 00:00:00.00000"
    else:
        fr_date_time = " 2020-01-01 00:00:00.00000"
    return fr_date_time


@app.post("/log_sync") 
async def syncLog(sync_data: str = Body(..., embed=True)): 
    global log_disconnect
    date = sync_data.split("\n")[0].split(", ")[2].split(" ")[0]
    
    file_path = os.path.join(LOG_DIR, f"log_{date}.csv")
    
    dstart = ""
    if not os.path.exists(file_path):
        print("Init file")
        dstart+="ID, Similarity, Time, State\n"
    if log_disconnect:
        print("Disconnected")
        dstart+="-, -, -, Sync not run\n"
        log_disconnect = False
    f = open(file_path, "a")
    f.write(dstart+sync_data)
    f.close()
    return {"msg":"Success"}