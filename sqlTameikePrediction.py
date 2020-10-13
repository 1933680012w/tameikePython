import datetime
import joblib
import keras 
import keras.callbacks
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import tensorflow
import xlrd
from keras import metrics
from keras.callbacks import TensorBoard
from keras.layers import LSTM,Dense
from keras.models import model_from_json,Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import fnSQL

before = 14

placeRows = fnSQL.sqlSelect("SELECT * FROM places")

for placeItem in placeRows:

    placeId = str(placeItem[0])

    waterLevelPredictionRowCount = fnSQL.sqlSelect("SELECT count(id) as count FROM waterLevelPredictions where place_id = " + placeId)

    lastInsertDateRows = fnSQL.sqlSelect("SELECT max(measurementDate) as maxMeasurementDate FROM waterLevelPredictions where place_id = " + placeId)
    sqlExecuteSentence = ""

    if lastInsertDateRows[0][0] == None:
        sqlExecuteSentence = "SELECT DATE_FORMAT(measurementDate, '%Y-%m-%d %H:00:00') AS time, "
        sqlExecuteSentence += "TRUNCATE(avg(waterLevel),0) as waterLevel, "
        sqlExecuteSentence += "TRUNCATE(avg(waterTemperature),2) as waterTemperature, "
        sqlExecuteSentence += "sum(precipitation) as precipitation, "
        sqlExecuteSentence += "TRUNCATE(avg(val1),2) as val1, "
        sqlExecuteSentence += "TRUNCATE(avg(val2),2) as val2, "
        sqlExecuteSentence += "TRUNCATE(avg(val3),2) as val3, "
        sqlExecuteSentence += "TRUNCATE(avg(val4),2) as val4 "
        sqlExecuteSentence += "FROM waterLevels "
        sqlExecuteSentence += "where place_id = " + placeId + " AND measurementDate < '" + datetime.datetime.now().strftime('%Y-%m-%d %H:00:00')  + "' "
        sqlExecuteSentence += "GROUP BY DATE_FORMAT(measurementDate, '%Y%m%d%H');"
    else :
        sqlExecuteSentence = "SELECT DATE_FORMAT(measurementDate, '%Y-%m-%d %H:00:00') AS time, "
        sqlExecuteSentence += "TRUNCATE(avg(waterLevel),0) as waterLevel, "
        sqlExecuteSentence += "TRUNCATE(avg(waterTemperature),2) as waterTemperature, "
        sqlExecuteSentence += "sum(precipitation) as precipitation, "
        sqlExecuteSentence += "TRUNCATE(avg(val1),2) as val1, "
        sqlExecuteSentence += "TRUNCATE(avg(val2),2) as val2, "
        sqlExecuteSentence += "TRUNCATE(avg(val3),2) as val3, "
        sqlExecuteSentence += "TRUNCATE(avg(val4),2) as val4 "
        sqlExecuteSentence += "FROM waterLevels "
        sqlExecuteSentence += "where place_id = " + placeId + " AND measurementDate >= '" + str(lastInsertDateRows[0][0]) + "' and measurementDate < '" + datetime.datetime.now().strftime('%Y-%m-%d %H:00:00')  + "' "
        sqlExecuteSentence += "GROUP BY DATE_FORMAT(measurementDate, '%Y%m%d%H');"

    insertRows = fnSQL.sqlSelect(sqlExecuteSentence)
    
    loopCount=0

    if waterLevelPredictionRowCount[0][0] <= 24 * 14 :

        for item in insertRows:
            measurementDate = datetime.datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S')  + datetime.timedelta(hours=1)

            measurementDate = str(item[0])

            sqlExecuteSentence = ""
            sqlExecuteSentencePreSentence = 'pre_1'
            
            for num in range(2, 25):
                sqlExecuteSentencePreSentence = sqlExecuteSentencePreSentence + ' ,pre_' + str(num)

            sqlInsertPredictWL = ''

            for num in range(1, 25):
                sqlInsertPredictWL += ",NULL as pre_" + str(num)

            sqlExecuteSentence ="INSERT INTO waterLevelPredictions(waterLevel,waterTemperature,precipitation,val1,val2,val3,val4,measurementDate,place_id,"
            sqlExecuteSentence += sqlExecuteSentencePreSentence +") "
            sqlExecuteSentence += "SELECT * FROM (SELECT '"
            sqlExecuteSentence += str(item[1])[:str(item[1]).find('.')] + "' as waterLevel,'" 
            sqlExecuteSentence += str(item[2]) + "' as waterTemperature,'"
            sqlExecuteSentence += str(item[3]) + "' as precipitation,'"
            sqlExecuteSentence += str(item[4]) + "' as val1,'" 
            sqlExecuteSentence += str(item[5]) + "' as val2 ,'" 
            sqlExecuteSentence += str(item[6]) + "' as val3 ,'"
            sqlExecuteSentence += str(item[7]) + "' as val4,'" 
            sqlExecuteSentence += str(measurementDate) + "' as measurementDate,"
            sqlExecuteSentence += placeId + " as place_id"
            sqlExecuteSentence += sqlInsertPredictWL
            sqlExecuteSentence += ") AS TEMP " + "WHERE NOT EXISTS (SELECT * FROM waterLevelPredictions WHERE measurementDate = '" + str(measurementDate) + "' AND place_id = " + placeId + ")"
            print(sqlExecuteSentence)
            fnSQL.sqlInsertUpdate(sqlExecuteSentence)
    else :
        for item in insertRows:
            measurementDate = datetime.datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S')  + datetime.timedelta(hours=1)

            measurementDate = str(item[0])
            wl = float(str(item[1])[:str(item[1]).find('.')] )/1000
            precipitation = float(str(item[3]))
            waterTemp = float(str(item[2]))
            
            lstmRows = fnSQL.sqlSelect("select * from(SELECT measurementDate,precipitation,waterLevel FROM `waterLevelPredictions` WHERE place_id = " + placeId + " AND  measurementDate <= '" + measurementDate + "' order by measurementDate DESC LIMIT " + str(before) + ") as A order by measurementDate asc")

            df = pd.DataFrame(lstmRows, columns=['time', 'precipitation', 'Water levels (m)'])
            
            predictPrecipitationRows = fnSQL.sqlSelect("SELECT predictTime,precipitation from predictPrecipitations WHERE place_id = " + placeId + " AND predictTime > '" + measurementDate + "' order by predictTime ASC LIMIT 24")
            dfPredictPrecipitation = pd.DataFrame(predictPrecipitationRows, columns=['time', 'precipitation'])
            
            dfPredictPrecipitation['Water levels (m)'] = df.iat[10, 2]
            df = df.append(dfPredictPrecipitation)

            df['Water levels (m)']=df['Water levels (m)'].astype(float)
            df['Water levels (m)']=df['Water levels (m)']/1000
            df['precipitation']=df['precipitation'].astype(float)

            df.insert(2, 'discharge', '0.0')
            df.insert(1, 'wind_speed', '0.0')
            df.insert(1, 'relative_humidity', '0.0')
            df.insert(1, 'ATPR', '0.0')
            df.insert(1, 'air_temperature', '0.0')
            df.insert(0, 'index', '0')
            
            df=df.reset_index()
            df = df.dropna(how='any')
            
            sc = joblib.load('/var/app/tameike/' + placeId + '_tameike_MinMax.pkl')
            
            model = model_from_json(open('/var/app/tameike/' + placeId + '_tameike_model'+'l'+'.json').read())
            model.load_weights('/var/app/tameike/' + placeId + '_tameike_model_weights'+'l'+'.h5')
            
            test = []
            now_data = []
            prediction = []
            
            test.append(sc.transform(np.array((pd.DataFrame(pd.concat([df[:38]['precipitation'],df[14:15]['Water levels (m)']])).reset_index(drop = True)).T)))
            test[0] = np.reshape(test[0],(test[0].shape[0],1,test[0].shape[1]))
            now_data.append(df['Water levels (m)'][14:15].values)
            prediction.append(now_data[0] - (pd.DataFrame(model.predict(test[0])).T))
            
            sqlExecuteSentence = ""
            sqlExecuteSentencePreSentence = 'pre_1'
            
            for num in range(2, 25):
                sqlExecuteSentencePreSentence = sqlExecuteSentencePreSentence + ' ,pre_' + str(num)

            sqlInsertPredictWL = ''
            
            if len(prediction[0]) >= 24:
                for index, row in prediction[0].iterrows():
                    
                    insertWaterLevel = str(round(float(row[0]*1000),0))
                    insertWaterLevel = insertWaterLevel[:insertWaterLevel.find('.')] 
                    sqlInsertPredictWL += ",'" + insertWaterLevel + "' as pre_" + str(index+1)
                    if index == 23 :
                        break

            sqlExecuteSentence ="INSERT INTO waterLevelPredictions(waterLevel,waterTemperature,precipitation,val1,val2,val3,val4,measurementDate,place_id,"
            sqlExecuteSentence += sqlExecuteSentencePreSentence +") "
            sqlExecuteSentence += "SELECT * FROM (SELECT '"
            sqlExecuteSentence += str(item[1])[:str(item[1]).find('.')] + "' as waterLevel,'" 
            sqlExecuteSentence += str(item[2]) + "' as waterTemperature,'"
            sqlExecuteSentence += str(item[3]) + "' as precipitation,'"
            sqlExecuteSentence += str(item[4]) + "' as val1,'" 
            sqlExecuteSentence += str(item[5]) + "' as val2 ,'" 
            sqlExecuteSentence += str(item[6]) + "' as val3 ,'"
            sqlExecuteSentence += str(item[7]) + "' as val4,'" 
            sqlExecuteSentence += str(measurementDate) + "' as measurementDate,"
            sqlExecuteSentence += placeId + " as place_id"
            sqlExecuteSentence += sqlInsertPredictWL
            sqlExecuteSentence += ") AS TEMP " + "WHERE NOT EXISTS (SELECT * FROM waterLevelPredictions WHERE measurementDate = '" + str(measurementDate) + "' AND place_id = " + placeId + ")"
            
            fnSQL.sqlInsertUpdate(sqlExecuteSentence)

            loopCount = loopCount + 1
            
            if(loopCount > 48):
                break