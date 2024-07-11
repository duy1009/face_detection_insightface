import cv2, glob, os, time, tqdm, shutil
from arcface import ArcFaceONNX
from logger import LogCSV

DB = "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/db_align"
MODEL_PATH = "/home/hungdv/tcgroup/Jetson/insightface/arc_R50.onnx"
IMG_DIR = "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/result_ang_duy/images"
LOG_RESULT_DIR = "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/result_ang_duy"

handler = ArcFaceONNX(MODEL_PATH)
handler.prepare(ctx_id=0)

# Load faces feature from database
db_img_path = glob.glob(os.path.join(DB, "**.png")) + glob.glob(os.path.join(DB, "**.jpg"))
feat_db = {}
for img_path in db_img_path:
    im = cv2.imread(img_path)
    name = os.path.basename(img_path)[:-4]
    feat_db[name] = handler.get2(im)

# Make dir to save result
result_dir = os.path.join(LOG_RESULT_DIR, "arcface")
print(f"[Result]: {result_dir}")
try:
    os.makedirs(os.path.join(result_dir, "stranger"))
except:
    pass
for feat_name, _ in feat_db.items():
    try:
        os.makedirs(os.path.join(result_dir, feat_name))
    except:
        pass

logg = LogCSV(os.path.join(result_dir, "result.csv"),
              ["Image", "Most_similar ", "Result", "Similarity", "Time_e2e(s)"]
              )

# Compare face
img_paths = glob.glob(os.path.join(IMG_DIR, "**.png")) + glob.glob(os.path.join(IMG_DIR, "**.jpg"))
for path in tqdm.tqdm(img_paths):
    st = time.time()
    img = cv2.imread(path)
    feat2 = handler.get2(img)
    score_max = -2
    feat_same = []
    for feat_name, feat in feat_db.items():
        score = handler.compute_sim(feat, feat2)
        if score > score_max:
            feat_same = [feat_name, score]
            score_max = score
    ms = feat_same[0]
    if feat_same[1] < 0.3:
        feat_same[0] = "stranger"
    path_save = os.path.join(result_dir, feat_same[0], os.path.basename(path))
    shutil.copy(path, path_save)
    logg.update([os.path.basename(path), ms, feat_same[0], feat_same[1], time.time()-st])
    logg.save()
    
    