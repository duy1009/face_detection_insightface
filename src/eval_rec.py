import cv2, glob, os, time, tqdm, shutil
import numpy as np
from arcface import ArcFaceONNX
from logger import LogCSV

MODEL_PATH = "/home/hungdv/tcgroup/Jetson/insightface/arc_R50.onnx"

# Images path on database
DB = ["/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/db/face_0.png",
      "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/db/face_8.png",
      "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/db/face_3.png",
      "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/db/face_6.png"
      ]

# Folders image test (The index of IMG_DIR must same the index of DB)
IMG_DIR = ["/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/face_images/cong/images",
           "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/face_images/duc/images",
           "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/face_images/duy/images"
           "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/face_images/hung/images"]

LOG_RESULT_DIR = "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/face_images"

N = 20


handler = ArcFaceONNX(MODEL_PATH)
handler.prepare(ctx_id=0)
nt = [0]*N
nf = [0]*N
step = 2/N

feats_db = [handler.get2(cv2.imread(i)) for i in DB]
# Make dir to save result
print(f"[Result]: {LOG_RESULT_DIR}")
try:
    os.makedirs(os.path.join(LOG_RESULT_DIR))
except:
    pass

logg_true = LogCSV(os.path.join(LOG_RESULT_DIR, "result_true.csv"),
              ["Image", "Similarity", "Time_e2e(s)"]
              )
logg_fales = LogCSV(os.path.join(LOG_RESULT_DIR, "result_false.csv"),
              ["Image", "Similarity", "Time_e2e(s)"]
              )

# Compare face
for ind_feat, feat in enumerate(feats_db):
    print(f"Eval Feature {ind_feat}:")
    for ind_dir, img_dir in tqdm.tqdm(enumerate(IMG_DIR)):
        img_paths = glob.glob(os.path.join(img_dir, "**.png")) + glob.glob(os.path.join(img_dir, "**.jpg"))
        for path in tqdm.tqdm(img_paths):
            st = time.time()
            img = cv2.imread(path)
            feat2 = handler.get2(img)
            score = handler.compute_sim(feat, feat2)
            ind = int((score+1)/step)
            if ind_feat==ind_dir:
                log = logg_true  
                nt[ind]+=1  
            else:
                log = logg_fales
                nf[ind]+=1  

            log.update([os.path.basename(path), score, time.time()-st])
            log.save()

log_s = LogCSV(os.path.join(LOG_RESULT_DIR, "result_final.csv"), ["Similarity", "Similar", "Dissimilar"])    
for i in range(N):
    val = -1 + i*step + step/2
    log_s.update([val, nt[i], nf[i]])
log_s.save()