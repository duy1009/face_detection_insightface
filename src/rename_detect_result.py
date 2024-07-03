from logger import LogCSV
import pandas as pd
import os.path as osp
import os
ROOT = r"/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/face_images/duc"
img_dir = osp.join(ROOT,"images")
detect_csv = osp.join(ROOT,"result.csv")
surfix = "_duc"
path_save = detect_csv
log = LogCSV(path_save, 
             ["Size_w", "Size_h", "Confident", "Time_e2e(s)", "Angle(degree)", "Path"]
             )

det = pd.read_csv(detect_csv).values
for det_res in det:
    old_path = det_res[5]
    name = osp.basename(old_path)
    old_path = osp.join(img_dir, name)
    new_name = name[:-4] + surfix + ".png"
    new_path = osp.join(img_dir, new_name)
    os.rename(old_path, new_path)
    log.update([det_res[0], 
                det_res[1], 
                det_res[2],
                det_res[3],
                det_res[4],
                new_path
                ])
    log.save()

