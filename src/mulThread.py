import cv2, time, os, datetime
import os.path as osp
from scrfd import SCRFD_INFERENCE
from tracker import CentroidTracker2
from _process import count_angle
from logger import LogCSV
import requests, io
from threading import Thread
from log_syn import run
from config import *
from utils import *
NP_buffer = []
processing_ids = []
enable_write = True
def requestRecognizeFace(image, id):
    global NP_buffer, processing_ids, enable_write
    processing_ids.append(id)
    image_byte = cv2.imencode(".jpg", image)[1].tobytes()
    files = {"image": io.BufferedReader(io.BytesIO(image_byte))}
    resp = requests.post(f'http://{HOST}:{PORT}/recognize_face', files=files)
    if resp.status_code == 200:
        data_recv = resp.json()
        enable_write = True
        NP_buffer.append([id, data_recv["name"], data_recv["score"]])
        enable_write = False
    processing_ids.remove(id)

def sendRequest(image, id):
    global processing_ids
    if len(processing_ids)<THREAD_REQUEST_MAX:
        thr = Thread(target=requestRecognizeFace, args=(image, id))
        thr.start()
    # thr.join()

def waitWrite():
    global enable_write
    while enable_write:
        continue
def updateNewPerson(IDs:list):
    '''NP_buffer: [length, (id, name, score)]
    '''
    IDs = IDs.copy()
    global NP_buffer
    now = time.time()
    while len(NP_buffer)>0:
        waitWrite()
        new_person = NP_buffer.pop(0)
        edit = False
        for _inx, _person in enumerate(IDs):
            if new_person[0] == _person[0]:
                if new_person[1] == "stranger":
                    new_person[1] = "stranger"+str(_person[0])
                if new_person[1] != _person[1]:
                    new_name = new_person[1]
                    log.update_a([f"{_person[1]} -> {new_name}", new_person[2], str(datetime.datetime.now()), "Change"])
                IDs[_inx] = [_person[0], new_person[1], time.time(), None, None, None, True]
                edit = True
                break
        
        if not edit:
            if new_person[1] == "stranger":
                new_person[1] = "stranger"+str(new_person[0])
            log.update_a([new_person[1], new_person[2], str(datetime.datetime.now()), "In"])
            IDs.append([new_person[0], new_person[1],  time.time(), None, None, None, True])
    return IDs

def syncLogThread():
    global kill_sync
    while not kill_sync:
        run()
        time.sleep(TIME_SYNC)

# Init
app = SCRFD_INFERENCE(model_path=MODEL_PATH)
app.prepare(ctx_id=0, input_size=DET_INPUT_SIZE)
vid = cv2.VideoCapture(SRC)
Track = CentroidTracker2(TRACK_MAX_HIDE_FRAME)

# Make save dir
temp_path = osp.join(LOG_DIR, "temp", "face.jpg")
if not osp.exists(osp.join(LOG_DIR, "temp")):
    os.makedirs(osp.join(LOG_DIR, "temp"))
path_log = osp.join(LOG_DIR, f"log_{datetime.date.today()}.csv")
if RESET_LOG and osp.exists(path_log):
    os.remove(path_log)
log = LogCSV(path = path_log,
             header = ["ID", "Similarity", "Time", "State"],
             mode = 'a'
             )
if not RESET_LOG:
    log.update_a(["-", 0, str(datetime.datetime.now()), "Reset device!"])

pre = time.time()
fcnt= 0
fps = 0
save_dis = False
IDs = []
pre_ids = []
pre_day = datetime.date.today()
kill_sync = False

sync_log_thr = Thread(target=syncLogThread)
sync_log_thr.start()
print("[Started log sync]")

while True:
    now = time.time()

    tempurature = int(os.popen('cat /sys/devices/virtual/thermal/thermal_zone0/temp').read())
    time_sleep = WARM_DELAY_TIME if tempurature > TEMPURATURE_WARM else NORMAL_DELAY_TIME
    print(f"Temp: {tempurature/1000} degree")
    if tempurature > TEMPURATURE_HOT:
        continue
    # Check FPS
    if now-pre>1:
        pre +=1
        fps = fcnt
        fcnt=0

    # Change save log file
    if pre_day != datetime.date.today():
        path_log = osp.join(LOG_DIR, f"log_{datetime.date.today()}.csv")
        log = LogCSV(path = path_log,
             header = ["ID", "Similarity", "Time", "State"],
             mode = 'a'
             )
    
    # Read image from the camera
    r, rimg = vid.read()
    if not r:
        break
    img_show = rimg.copy()

    # Detect face
    pred = app.detect(rimg)
    faces_r = postprocess(pred)

    # Filter bad faces
    faces = filter(faces_r, DET_CONF_FIL, FACE_ANGLE_FIL)
    
    # Tracking faces
    res = Track.update(faces)
    
    IDs = updateNewPerson(IDs)
    # re-recognition
    ids = []
    for _inx, (_id,_name, _t, _img_align, _img_size, _angle, _block) in enumerate(IDs):
        time_request = TIME_REQUEST_NSTRANGER if len(_name) >8 and _name[:8] != "stranger" else TIME_REQUEST_STRANGER
        if  now - _t > time_request and not _block and _id not in processing_ids:
            if (imgs_size[0][0] + imgs_size[0][1])/2 > IMG_SIZE_REQUEST_MIN:
                print("[Request]: re-recognition")
                sendRequest(_img_align, _id)
                IDs[_inx][2] = now
        ids.append(_id)

    # Person in  
    track_ids = [] 
    for track_res in res:
        for _inx, (_id,_name, _t, _img_align, _img_size, _angle, _block) in enumerate(IDs):
            # Update better image
            if track_res["id"] == _id:
                if _img_align is None or trackIsBetter(track_res, IDs[_inx], SCORE_RATE):
                    imgs_f, imgs_size = alignCrop(rimg, [track_res])
                    IDs[_inx][3] = imgs_f[0]
                    IDs[_inx][4] = imgs_size[0]
                    IDs[_inx][5] = count_angle(track_res["kps"])
                    IDs[_inx][6] = False
        # Check new person
        if track_res["id"] not in ids + processing_ids:
            imgs_f, imgs_size = alignCrop(rimg, [track_res])
            if (imgs_size[0][0] + imgs_size[0][1])/2 > IMG_SIZE_REQUEST_MIN:
                print("[Request]: new person")
                sendRequest(imgs_f[0], track_res["id"])
        track_ids.append(track_res["id"])
    
    # Person out
    ids_now = [i[0] for i in IDs]
    for _id, _name in pre_ids:
        if _id not in ids_now:
            log.update_a([_name,"-", str(datetime.datetime.now()), "Out"])
    pre_ids = [(i[0], i[1]) for i in IDs]        

    ids_temp = []
    for _ids in IDs: 
        if _ids[0] in track_ids:
            ids_temp.append(_ids)
    IDs = ids_temp
    fcnt+=1
    if SHOW_DETECT:
        img_show = app.draw_on(img_show, faces_r)
    if SHOW_RECOGNIZE:
        img_show = drawFace(img_show, res, IDs)
    cv2.putText(img_show, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (170, 0, 0), 1)
    cv2.imshow("Faces detection", cv2.resize(img_show, IMG_SIZE_SHOW))
    k = cv2.waitKey(time_sleep)
    if k == ord("q"):
        break
    # break

kill_sync = True
cv2.destroyAllWindows()