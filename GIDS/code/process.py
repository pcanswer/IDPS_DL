import csv
import numpy as np
import math
import torch

can_num = 48
feature_num = 48
with open('../data/gear_dataset.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
column1 = [row[1] for row in rows]
CAN_ID = [x[1:] for x in column1]
label_raw = [row[-1] for row in rows]

hexadecimal = '1234567890abcdef'
def one_hot(data):
       char_to_int = dict((c, i) for i, c in enumerate(hexadecimal))
       int_to_char = dict((i, c) for i, c in enumerate(hexadecimal))
       integer_encoded = [char_to_int[char] for char in data]
       onehot_encoded = list()
       for value in integer_encoded:
              letter = [0 for _ in range(len(hexadecimal))]
              letter[value] = 1
              onehot_encoded.append(letter)
       return onehot_encoded

hexadecimal_num = '0123456789abcdef'
def encode(data):
    char_to_int = dict((c, i) for i, c in enumerate(hexadecimal_num))
    integer_encoded = [char_to_int[char] for char in data]
    return integer_encoded


one_hot_set=[]
for i in CAN_ID:
       CAN_ID2one_hot = one_hot(i)
       one_hot_set.append(CAN_ID2one_hot)
# one_hot_set
one_hot_set = np.array(one_hot_set)
# print('one_hot_set: ',one_hot_set)

num = math.floor(len(one_hot_set)/can_num)
count = 0
image = []
for count in range(num):
    b = one_hot_set[count:count+can_num]
    image.append(b.reshape(can_num,feature_num))
    count += can_num
label_l2n = []
for i in label_raw:
    if i == 'R':
        label_int = 0
    else:
        label_int = 1
    label_l2n.append(label_int)

count = 0
label_sec = []
for count in range(num):
    label_set = label_l2n[count:count+can_num]
    label_sec.append(label_set)
    count +=can_num

label = []

for i in label_sec:
    a = sum(i)
    if a == 0:
        label.append(1)
    else:
        label.append(0)

num_attackimage = sum(label)

np.save("../data/dcgan/gear_image_data.npy",image)
np.save('../data/dcgan/gear_image_label.npy',label)