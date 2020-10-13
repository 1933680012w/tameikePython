import datetime
import json
import numpy as np
import os
import pandas as pd
import requests
import time
import fnSQL

placeRows = fnSQL.sqlSelect("SELECT * FROM places")
yumakeKey = fnSQL.sqlSelect("SELECT yumakeKey from settings where id = 1")[0][0]

for item in placeRows:
    lastPredictTimeRows = fnSQL.sqlSelect("SELECT MAX(predictTime) FROM predictPrecipitations where place_id = " + str(item[0]))
    arr_lastPredictTime = np.array(lastPredictTimeRows)
    lastPredictTime = np.transpose(arr_lastPredictTime[0])

    print(lastPredictTime)
    url = "http://api.yumake.jp/1.0/forecastMsm.php?lat=" + str(item[4]) + "&lon=" + str(item[3]) + "&key=" + yumakeKey

    getDataFlg = False

    for i in range(10):
        r = requests.get(url)
        data = r.json()

        if data is None :
            getDataFlg = False
        else:
            if 0 in data :
                getDataFlg = False
            else :
                if str(data['status']) == 'error':
                    getDataFlg = False
                elif str(data['status']) == 'success':
                    getDataFlg = True
                    df = pd.DataFrame(data["forecast"])
                    df = df.drop("windSpeed", axis=1)
                    df = df.drop("windDir", axis=1)
                    df = df.drop("windDirStr", axis=1)
                    df = df.drop("meanSeaLevelPressure", axis=1)
                    df = df.drop("temperature", axis=1)
                    df = df.drop("relativeHumidity", axis=1)
                    df = df.drop("lcdc", axis=1)
                    df = df.drop("mcdc", axis=1)
                    df = df.drop("hcdc", axis=1)
                    df = df.drop("tcdc", axis=1)
                    break
        print('False')
        time.sleep(5)

    if getDataFlg == True :
        df['forecastDateTime'] = pd.to_datetime(df['forecastDateTime'])
        df['predictDate'] = df['forecastDateTime'].dt.strftime('%Y/%m/%d %H:%M')
        df = df[1:]
        for index, row in df.iterrows():
            if datetime.datetime.now()  > datetime.datetime.strptime(row[2], '%Y/%m/%d %H:%M'):
                pass
            elif datetime.datetime.strptime(row[2], '%Y/%m/%d %H:%M') <=  lastPredictTime[0]:
                fnSQL.sqlInsertUpdate('update predictPrecipitations set precipitation = "'  + str(row[1]) + '",modified = "' + datetime.datetime.now().strftime('%Y/%m/%d %H:%M:00') + '" where place_id = ' + str(item[0]) + ' and predictTime = "'+str(row[2])+':00"')
            else :
                fnSQL.sqlInsertUpdate('insert into predictPrecipitations(predictTime,precipitation,place_id) values ("' + str(row[2]) + ':00","'  + str(row[1]) + '", ' + str(item[0]) + ') ')
    else:
        print('APIFalse')
    time.sleep(10)
    




