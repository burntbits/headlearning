import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
#data = pd.DataFrame(columns=7)
sort_key = lambda s:(len(s), s)
accel_headers = ['time','acc_X', 'acc_Y', 'acc_Z']
gyro_headers = ['time','gyro_X', 'gyro_Y', 'gyro_Z']
ir_headers = ['time','prox']
final_headers = ['acc_X', 'acc_Y', 'acc_Z', 'gyro_X', 'gyro_Y', 'gyro_Z', 'prox', 'label']
finaldata = pd.DataFrame(columns = final_headers)
scaler = MinMaxScaler()
for dirs in next(os.walk('.'))[1]:
    sorted_dirs = os.listdir(dirs)
    sorted_dirs.sort(key=sort_key)
    print (sorted_dirs)
    num_files = int(len(sorted_dirs)/3)
    accel_data = np.empty(3)
    gyro_data = np.empty(3)
    ir_data = np.empty(1)
    for i in range(0,num_files):
        accel_csv = (pd.read_csv(dirs + '/accel_data' + str(i) + '.csv', names=accel_headers)).drop('time', axis=1)
        accel_data = np.vstack((accel_data,scaler.fit_transform(accel_csv)))
        gyro_csv = (pd.read_csv(dirs + '/gyro_data' + str(i) + '.csv', names=gyro_headers)).drop('time', axis=1)
        gyro_data = np.vstack((gyro_data,scaler.fit_transform(gyro_csv)))
        ir_csv = (pd.read_csv(dirs + '/ir_data' + str(i) + '.csv', names=ir_headers)).drop('time', axis=1)
        ir_data = np.vstack((ir_data,scaler.fit_transform(ir_csv)))
    if (dirs=='Mahesh VC'):
        data_indiv = np.hstack((accel_data,gyro_data,ir_data,np.ones((len(accel_data),1),dtype=np.int8)))
    else:
        #if (np.shape(accel_data)!=np.shape(gyro_data)):
        #   accel_data = np.delete(accel_data, [-1])
        data_indiv = np.hstack((accel_data,gyro_data,ir_data,np.zeros((len(accel_data),1),dtype=np.int8)))
    data = pd.DataFrame(data_indiv, columns=final_headers)
    finaldata = pd.concat([data,finaldata])
finaldata.to_csv('finaldata.csv', index=False)
'''
x=pd.read_csv('Prasanth/final.csv')
y=pd.read_csv('Shahabas/final.csv')
x = pd.concat([x,y])
arr = x.to_numpy()
#arr.reshape(shape)
x.to_csv('finalir.csv', index=False)'''