from logger import LogCSV
import pandas as pd
import os.path as osp
detect_csv = "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/face_images/duc/result.csv"
recog_csv = "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/face_images/duc/arcface/result.csv"
path_save = "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/face_images/duc/final_resut.csv"
log = LogCSV(path_save, 
             ["Image", "Size_w", "Size_h", "Confident", "Angle(degree)", "Detect_time(s)", "Most_similar", "Similarity", "Result", "Recognition_time(s)"]
             )

det = pd.read_csv(detect_csv).values
rec = pd.read_csv(recog_csv).values
for det_res in det:
    name = osp.basename(det_res[5])
    rec_res = None
    for r in rec:
        if r[0] == name:
            rec_res = r
            break
    if rec_res is not None:
        log.update([name, 
                    det_res[0], 
                    det_res[1], 
                    det_res[2],
                    det_res[4],
                    det_res[3],
                    rec_res[1],
                    rec_res[3],
                    rec_res[2],
                    rec_res[4]
                    ])
        log.save()

