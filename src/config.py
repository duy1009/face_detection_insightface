# API
HOST = '192.168.2.233'
PORT = '8000'

# Path
LOG_DIR = "log" 
MODEL_PATH = "weights/det_10g.onnx"
# SRC = "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/src/IMG_5272.MOV"
LOG_MODE = 1
SRC = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)3264, height=(int)2464, framerate=(fraction)21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)3264, height=(int)2464, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=True"

# Config time
TIME_REQUEST_NSTRANGER= 10  
TIME_REQUEST_STRANGER = 5
RESET_LOG = True

# Config face detection
DET_INPUT_SIZE = (640, 640)
DET_CONF_FIL = 0.5
FACE_ANGLE_FIL = 23

# Config tracking
TRACK_MAX_HIDE_FRAME = 10
SCORE_RATE = 0.5

# Config recognize
IMG_SIZE_REQUEST_MIN = 5
THREAD_REQUEST_MAX = 10

# Log sync
TIME_SYNC = 3

# Tempurature
TEMPURATURE_WARM = 50000 
TEMPURATURE_HOT = 80000
WARM_DELAY_TIME = 100
NORMAL_DELAY_TIME = 50

# Show
IMG_SIZE_SHOW = (1000, 800)
SHOW_RECOGNIZE = True
SHOW_DETECT = True