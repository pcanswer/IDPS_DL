import pandas as pd
import numpy as np


def str2hex(s):
    if type(s) is not str:
        return s
    odata = 0;
    su =s.upper()
    for c in su:
        tmp=ord(c)
        if tmp <= ord('9') :
            odata = odata << 4
            odata += tmp - ord('0')
        elif ord('A') <= tmp <= ord('F'):
            odata = odata << 4
            odata += tmp - ord('A') + 10
    return odata

df = pd.read_csv('../data/Fuzzy_dataset.csv')

df.fillna(0,inplace=True)
print(df)
df_data = pd.DataFrame(columns=('time','canid','length','byte0','byte1','byte2','byte3','byte4','byte5','byte6','byte7','bit0','bit1','bit2','bit3','bit4','bit5','bit6','bit7','bit8','bit9','bit10','bit11','bit12','bit13','bit14','bit15','bit16','bit17','bit18','bit19','bit20','bit21','bit22','bit23','bit24','bit25','bit26','bit27','bit28','bit29','bit30','bit31','bit32','bit33','bit34','bit35','bit36','bit37','bit38','bit39','bit40','bit41','bit42','bit43','bit44','bit45','bit46','bit47','bit48','bit49','bit50','bit51','bit52','bit53','bit54','bit55','bit56','bit57','bit58','bit59','bit60','bit61','bit62','bit63','label'))
last_msg_time = {}

def bin_process(number,index):
    return (number& pow(2,index)) >> index

for index in df.index:
    if((index % 200) ==0):
        print(index)
        if(index > 25000):
            break

    df.iloc[index, 1] = str2hex(df.iloc[index, 1])
    for i in range(df.iloc[index,2]):
        df.iloc[index, 3 + i] = str2hex(df.iloc[index,3+i])
    if(df.iloc[index, 2] != 8):
        df.iloc[index,11] = df.iloc[index,df.iloc[index,2]+3]
        df.iloc[index, df.iloc[index, 2] + 3] = 0

#delta time
    if df.iloc[index, 1] not in last_msg_time.keys():
        last_msg_time[df.iloc[index, 1]] = df.iloc[index, 0]
        df.iloc[index,0] = 0
    else:
        tmp = df.iloc[index, 0]
        df.iloc[index, 0] = df.iloc[index,0] - last_msg_time[df.iloc[index, 1]]
        last_msg_time[df.iloc[index, 1]] = tmp

    tmp_list = []
    tmp_list.append(df.iloc[index, 0])
    tmp_list.append(df.iloc[index, 1])
    tmp_list.append(df.iloc[index, 2])
    for i in range(8):
        tmp_list.append(df.iloc[index,3+i])
    for i in range(8):
        for j in range(7,-1,-1):
            tmp_list.append(bin_process(df.iloc[index,3+i],j))
    tmp_list.append(df.iloc[index,11])
    df_data.loc[index] = tmp_list

df_data.to_csv("../data/Fuzz_data_.csv")
print(df_data)