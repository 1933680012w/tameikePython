import datetime
import joblib
import keras 
import keras.callbacks
import keras.backend as K
import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd 
import random
import tensorflow
import xlrd
from keras import metrics
from keras.callbacks import TensorBoard
from keras.layers import LSTM,SimpleRNN,GRU,Dense
from keras.layers.core import Activation, Dense,Flatten,Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import fnSQL

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

os.makedirs('/var/app/tameike', exist_ok=True)

nowHour = datetime.datetime.now().time().hour

placeId = str(nowHour + 1)

before = 14
Output_set = before + 23
precipitation_set = before + 24
Input_shape= before + 25

rows = fnSQL.sqlSelect("SELECT measurementDate,precipitation,waterLevel FROM waterLevelPredictions where place_id = " + placeId + " AND length(precipitation) > 0 and length(waterLevel) > 0 order by measurementDate ASC")

#ColumnNameAdd
df = pd.DataFrame(rows, columns=['time', 'precipitation', 'Water levels (m)'])
df["precipitation"]=df["precipitation"].astype(float)
df["Water levels (m)"]=df["Water levels (m)"].astype(float)
df["Water levels (m)"]=df["Water levels (m)"]/1000

#columnAdd
df.insert(1, 'wind_speed', '0.0')
df.insert(1, 'relative_humidity', '0.0')
df.insert(1, 'ATPR', '0.0')
df.insert(1, 'air_temperature', '0.0')
df.insert(0, 'index', '0')

#dtypeConvert
df["wind_speed"]=df["wind_speed"].astype(float)
df["relative_humidity"]=df["relative_humidity"].astype(float)
df["ATPR"]=df["ATPR"].astype(float)
df["air_temperature"]=df["air_temperature"].astype(float)
df["index"]=df["index"].astype(int)

df = df.dropna(how='any')

rowLen=len(df)

if rowLen > 288 :

    rowLen=int(rowLen/4)

    timeline = np.arange(rowLen)

    df1 = df[:rowLen].reset_index().drop(['level_0'],1)
    df2 = df[(rowLen + 1):(rowLen*2)].reset_index().drop(['level_0'],1)
    df3 = df[((rowLen*2) + 1):(rowLen*3)].reset_index().drop(['level_0'],1)
    df4 = df[((rowLen*3)+1):].reset_index().drop(['level_0'],1)

    L1 = len(df1)
    L2 = len(df2)
    L3 = len(df3)
    L4 = len(df4)

    mat = np.zeros([24,8])
    mat[:,:] = np.nan
    mat = pd.DataFrame(mat)
    mat.columns = ['index','time','air_temperature','ATPR','relative_humidity','wind_speed','precipitation','Water levels (m)']
    df1_1 = pd.concat([df1,mat]).reset_index()
    df2_1 = pd.concat([df2,mat]).reset_index()
    df3_1 = pd.concat([df3,mat]).reset_index()
    df4_1 = pd.concat([df4,mat]).reset_index()

    WL_1 = []
    for i in range(0,25):
        WL_1.append(df1_1['Water levels (m)'].shift(i))
    
    

    Out_put_1 = pd.DataFrame(WL_1,index = [i for i in range(0,25)])
    Out_put_1 = Out_put_1.drop([i for i in range(0,24)],1).drop([i for i in range(L1,L1+24)],1).iloc[::-1].reset_index().drop(['index'],1).T.reset_index().drop(['index'],1)
    Out_put_1 = pd.concat([df1['time'],df1['Water levels (m)'],Out_put_1],1).dropna(how = 'any')

    WL_2 = []
    for i in range(0,25):
        WL_2.append(df2_1['Water levels (m)'].shift(i))

    Out_put_2 = pd.DataFrame(WL_2,index = [i for i in range(0,25)])
    Out_put_2 = Out_put_2.drop([i for i in range(0,24)],1).drop([i for i in range(L2,L2+24)],1).iloc[::-1].reset_index().drop(['index'],1).T.reset_index().drop(['index'],1)
    Out_put_2 = pd.concat([df2['time'],df2['Water levels (m)'],Out_put_2],1).dropna(how = 'any')

    WL_3 = []
    for i in range(0,25):
        WL_3.append(df3_1['Water levels (m)'].shift(i))
        
    Out_put_3 = pd.DataFrame(WL_3,index = [i for i in range(0,25)])
    Out_put_3 = Out_put_3.drop([i for i in range(0,24)],1).drop([i for i in range(L3,L3+24)],1).iloc[::-1].reset_index().drop(['index'],1).T.reset_index().drop(['index'],1)
    Out_put_3 = pd.concat([df3['time'],df3['Water levels (m)'],Out_put_3],1).dropna(how = 'any')

    WL_4 = []
    for i in range(0,25):
        WL_4.append(df4_1['Water levels (m)'].shift(i))
        
    Out_put_4 = pd.DataFrame(WL_4,index = [i for i in range(0,25)])
    Out_put_4 = Out_put_4.drop([i for i in range(0,24)],1).drop([i for i in range(L4,L4+24)],1).iloc[::-1].reset_index().drop(['index'],1).T.reset_index().drop(['index'],1)
    Out_put_4 = pd.concat([df4['time'],df4['Water levels (m)'],Out_put_4],1).dropna(how = 'any')

    Out_put_1 = Out_put_1.drop([i for i in range(0,Output_set)]).reset_index().drop(['index'],1)
    Out_put_2 = Out_put_2.drop([i for i in range(0,Output_set)]).reset_index().drop(['index'],1)
    Out_put_3 = Out_put_3.drop([i for i in range(0,Output_set)]).reset_index().drop(['index'],1)
    Out_put_4 = Out_put_4.drop([i for i in range(0,Output_set)]).reset_index().drop(['index'],1)

    for i in range(1,25):
        Out_put_1[i] = Out_put_1[0]-Out_put_1[i]
        Out_put_2[i] = Out_put_2[0]-Out_put_2[i]
        Out_put_3[i] = Out_put_3[0]-Out_put_3[i]
        Out_put_4[i] = Out_put_4[0]-Out_put_4[i]

    PRECIPITATION_1 = []
    for i in range(0,precipitation_set):
        PRECIPITATION_1.append(df1['precipitation'].shift(i))

    In_put_1 = pd.DataFrame(PRECIPITATION_1,index = [i for i in range(0,precipitation_set)])
    In_put_1 = In_put_1.dropna(how = 'any',axis = 1).iloc[::-1].reset_index().drop(['index'],1).T.reset_index().drop(['index'],1).drop([i for i in range(0,24)]).reset_index().drop(['index'],1)

    PRECIPITATION_2 = []

    for i in range(0,precipitation_set):
        PRECIPITATION_2.append(df2['precipitation'].shift(i))
        
    In_put_2 = pd.DataFrame(PRECIPITATION_2,index = [i for i in range(0,precipitation_set)])
    In_put_2 = In_put_2.dropna(how = 'any',axis = 1).iloc[::-1].reset_index().drop(['index'],1).T.reset_index().drop(['index'],1).drop([i for i in range(0,24)]).reset_index().drop(['index'],1)

    PRECIPITATION_3 = []

    for i in range(0,precipitation_set):
        PRECIPITATION_3.append(df3['precipitation'].shift(i))

    In_put_3 = pd.DataFrame(PRECIPITATION_3,index = [i for i in range(0,precipitation_set)])
    In_put_3 = In_put_3.dropna(how = 'any',axis = 1).iloc[::-1].reset_index().drop(['index'],1).T.reset_index().drop(['index'],1).drop([i for i in range(0,24)]).reset_index().drop(['index'],1)

    PRECIPITATION_4 = []

    for i in range(0,precipitation_set):
        PRECIPITATION_4.append(df4['precipitation'].shift(i))

    In_put_4 = pd.DataFrame(PRECIPITATION_4,index = [i for i in range(0,precipitation_set)])
    In_put_4 = In_put_4.dropna(how = 'any',axis = 1).iloc[::-1].reset_index().drop(['index'],1).T.reset_index().drop(['index'],1).drop([i for i in range(0,24)]).reset_index().drop(['index'],1)

    In_put_6 = pd.concat([In_put_1,In_put_2,In_put_3,In_put_4]).reset_index(drop = True)
    In_put_7 = pd.concat([Out_put_1['Water levels (m)'],Out_put_2['Water levels (m)'],Out_put_3['Water levels (m)'],Out_put_4['Water levels (m)']]).reset_index(drop = True)

    INPUT = pd.concat([In_put_6,In_put_7],1)
    OUTPUT = pd.concat([Out_put_1,Out_put_2,Out_put_3,Out_put_4]).reset_index(drop = True).drop(['time','Water levels (m)',0],1)

    INPUT = np.array(INPUT)
    sc= MinMaxScaler()
    sc.fit(INPUT)
    INPUT = sc.transform(INPUT)
    INPUT = np.reshape(INPUT,(INPUT.shape[0],1,INPUT.shape[1]))
    OUTPUT = np.array(OUTPUT)

    model = Sequential()
    model.add(LSTM(32,input_shape=(1, Input_shape)))
    model.add(Dense(24))

    model.compile(loss='mean_absolute_error',optimizer = 'Adam',metrics=[rmse])

    history = model.fit(INPUT,OUTPUT,epochs=300,verbose=2,batch_size = 64)

    g = 'model_json_str'+'l'
    g = model.to_json()

    open('/var/app/tameike/' + placeId + '_tameike_model'+'l'+'.json', 'w').write(g)
    model.save_weights('/var/app/tameike/' + placeId + '_tameike_model_weights'+'l'+'.h5')
    joblib.dump(sc, '/var/app/tameike/' + placeId + '_tameike_MinMax.pkl') 
