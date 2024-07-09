# API
HOST = '192.168.2.233'
PORT = '8000'

# Path
LOG_DIR = "log" 
MODEL_PATH = "weights/det_10g.onnx"
SRC = "/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/src/IMG_5272.MOV"

# Config time
TIME_REQUEST_NSTRANGER= 10  
TIME_REQUEST_STRANGER = 5
RESET_LOG = True

# Config face detection
DET_INPUT_SIZE = (640, 640)
DET_CONF_FIL = 0.5
FACE_ANGLE_FIL = 180

# Config tracking
TRACK_MAX_HIDE_FRAME = 10
SCORE_RATE = 0.5

# Log sync
TIME_SYNC = 3