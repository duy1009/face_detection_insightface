from fastapi import FastAPI, Body
import datetime, os, glob
import pandas as pd

LOG_DIR = "log"
def convertStr2Time(str):
    return datetime.datetime.strptime(str, ' %Y-%m-%d %H:%M:%S.%f')

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

app = FastAPI() 

@app.get("/last_time_log") 
def syncLastTime(): 
    # contents = image.file.read()
    files = glob.glob(os.path.join(LOG_DIR, "**.csv"))
    pre_date = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d").date()
    empty = True
    for file in files:
        f_date = datetime.datetime.strptime(os.path.basename(file)[4:-4], "%Y-%m-%d").date()
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
    date = sync_data.split("/n")[0].split(", ")[2].split(" ")[0]
    file_path = os.path.join(LOG_DIR, f"log_{date}.csv")
    f = open(file_path, "a")
    f.write(sync_data)
    f.close()
    return {"msg":"Success"}


