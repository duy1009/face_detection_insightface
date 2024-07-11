import pandas as pd
import datetime
import requests, os, glob, time
from config import HOST, PORT, LOG_DIR, LOG_MODE

def getLastTime(): 
    # contents = image.file.read()
    resp = requests.get(f'http://{HOST}:{PORT}/last_time_log')
    if resp.status_code == 200:
        data_recv = resp.json()
        return data_recv
def convertStr2Time(str):
        return datetime.datetime.strptime(str, ' %Y-%m-%d %H:%M:%S.%f')

def syncLog(last_time):
    files = glob.glob(os.path.join(LOG_DIR, "**.csv"))
    log_paths = []
    for f in files:
        try:
            date = datetime.datetime.strptime(os.path.basename(f)[4:-4], "%Y-%m-%d").date()
        except:continue
        if date>=last_time.date():
            log_paths.append(f)
    for log_path in log_paths:
    # log_path = os.path.join(LOG_DIR, f"log_{str(last_time.date())}.csv")
        df = pd.read_csv(log_path).values

        pre_date_time = last_time
        index = 0
        sync = False
        for ind, frame in enumerate(df):
            if convertStr2Time(frame[2]) > pre_date_time:
                index = ind
                sync = True
                break
        if sync:
            data = ""
            for frame in df[index:]:
                frame1 =  (" "+str(frame[1])) if type(frame[1]) == float else frame[1]
                if LOG_MODE == 0:
                    data+=f"{frame[0]},{frame1},{frame[2]},{frame[3]}\n"
                else:
                    data+=f"{frame[0]},{frame1},{frame[2]}\n"
            
            resp = requests.post(f'http://{HOST}:{PORT}/log_sync', json={"sync_data":data})
            if resp.status_code == 200:
                print("[Sync log]: "+resp.json()["msg"])
            else:
                print(f"[Sync log]: Error code {resp.status_code}")

    
kill_sync = False
def run():
    last_time_str = getLastTime()
    last_time = convertStr2Time(last_time_str)
    syncLog(last_time)
