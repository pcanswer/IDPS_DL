import csv
import pandas as pd


last_msg_time = {}
df_data = pd.DataFrame(columns=('time','canid','length','byte0','byte1','byte2','byte3','byte4','byte5','byte6','byte7','label'))

with open('../data/gear_dataset.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

index = 0
for row in rows:
    row[0] = float(row[0])
    row[2] = int(row[2])
    if row[1] not in last_msg_time.keys():
        last_msg_time[row[1]] = row[0]
        row[0] = 0
    else:
        tmp = row[0]
        row[0] = row[0] - last_msg_time[row[1]]
        last_msg_time[row[1]] = tmp
    tmp_list = []
    tmp_list.append(row[0])
    tmp_list.append(int(row[1],16))
    tmp_list.append(row[2])
    for i in range(row[2]):
        tmp_list.append(int(row[3+i],16))
    for i in range(8-row[2]):
        tmp_list.append(0)
    if row[-1] == 'T':
        row[-1] = 'G'
    tmp_list.append(row[-1])
    df_data.loc[index] = tmp_list
    index = index + 1
    if((index % 200) ==0):
        print(index)
    if(index > 50000):
        break


df_data.to_csv("../data/gear_data_with_payload.csv")
print(df_data)