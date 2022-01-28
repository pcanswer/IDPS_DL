try:
    import scapy.all as scapy
except ImportError:
    import scapy
from scapy.utils import PcapReader
import binascii
import xlsxwriter
import torch
from torch import nn
import torchvision
import torch.utils.data as Data
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

def xlswrite(worksheet, row, msg):
    worksheet.write("A"+str(row), msg['service_id'])
    worksheet.write("B"+str(row), msg['method_id']) 
    worksheet.write("C"+str(row), msg['client_id']) 
    worksheet.write("D"+str(row), msg['message_type']) 
    worksheet.write("E"+str(row), msg['session_id']) 
    worksheet.write("F"+str(row), msg['interface_version']) 
    worksheet.write("G"+str(row), msg['protocol_version']) 
    worksheet.write("H"+str(row), msg['return_code']) 
    worksheet.write("I"+str(row), msg['ip_src']) 
    worksheet.write("J"+str(row), msg['ip_dest']) 
    worksheet.write("K"+str(row), msg['protocol']) 
    worksheet.write("L"+str(row), msg['port_src']) 
    worksheet.write("M"+str(row), msg['port_dest']) 
    worksheet.write("N"+str(row), msg['mac_src'])
    worksheet.write("O"+str(row), msg['mac_dest'])
    worksheet.write("P" + str(row), msg['label'])
    worksheet.write("Q" + str(row), msg['time'])

def is_sequence(msg1,msg2):
    if msg1['service_id']==msg2['service_id'] and msg1['method_id']==msg2['method_id'] and msg1['client_id']==msg2['client_id'] and msg1['session_id']==msg2['session_id'] and ((msg1['ip_src']==msg2['ip_dest'] and msg1['ip_dest']==msg2['ip_src']) or((msg1['ip_src']==msg2['ip_src'] and msg1['ip_dest']==msg2['ip_dest']))):
        return True
    else:
        return False

def one_hot_encode(raw_list,param):
    tmp_list=[]
    for i in range(len(raw_list)):
        # tmp_list.append(index_list.index(raw_list[i][param]))
        tmp_list.append(raw_list[i][param])
    return lb.fit_transform(tmp_list)

#return X Y features_input
def data_handle(xlsx_name, pcap_name):
    workbook = xlsxwriter.Workbook(xlsx_name)
    worksheet = workbook.add_worksheet()
    worksheet.write("A1", 'service_id')
    worksheet.write("B1", 'method_id')
    worksheet.write("C1", 'client_id')
    worksheet.write("D1", 'message_type')
    worksheet.write("E1", 'session_id')
    worksheet.write("F1", 'interface_version')
    worksheet.write("G1", 'protocol_version')
    worksheet.write("H1", 'return_code')
    worksheet.write("I1", 'ip_src')
    worksheet.write("J1", 'ip_dest')
    worksheet.write("K1", 'protocol')
    worksheet.write("L1", 'port_src')
    worksheet.write("M1", 'port_dest')
    worksheet.write("N1", 'mac_src')
    worksheet.write("O1", 'mac_dest')
    worksheet.write("P1", 'label')
    worksheet.write("Q1", 'time')
  
    packets = scapy.rdpcap(pcap_name)
    time=0
    msg_list=[]

    for data in packets:
        msg={}
        raw_data=str(binascii.b2a_hex(bytes(data)))[2:-1]
        msg['mac_dest'] = raw_data[0:12]
        msg['mac_src'] = raw_data[12:24]
        msg['protocol'] = raw_data[46:48]
        msg['ip_src'] = raw_data[52:60]
        msg['ip_dest'] = raw_data[60:68]
        msg['port_src'] = raw_data[68:72]
        msg['port_dest'] = raw_data[72:76]
        msg['service_id'] = raw_data[84:88]
        msg['method_id'] = raw_data[88:92]
        msg['client_id'] = raw_data[100:104]
        msg['session_id'] = raw_data[104:108]
        msg['protocol_version'] = raw_data[108:110]
        msg['interface_version'] = raw_data[110:112]
        msg['message_type'] = raw_data[112:114]
        msg['return_code'] = raw_data[114:116]
        msg['label'] = 0

        msg['time'] = time
        time=time+1
        msg_list.append(msg)

    #攻击报文标注
    del_response_count=0
    del_request_count=0
    for i in range(len(msg_list)):
        find_response = 0
        find_request = 0
        if msg_list[i]['message_type'] == "00":
            for j in range(i+1,len(msg_list)):
                if is_sequence(msg_list[j],msg_list[i]):
                    if msg_list[j]['message_type'] == "00":
                        break
                    if msg_list[j]['message_type'] == "80" or msg_list[j]['message_type'] == "81":
                        find_response = 1
                        break
            if find_response == 0:
                del_response_count = del_response_count+1
                msg_list[i]['label'] = 1


        elif msg_list[i]['message_type']  == "80" or msg_list[i]['message_type']  == "81":
            for j in range(i,0,-1):
                j=j-1
                if is_sequence(msg_list[j],msg_list[i]):
                    if (msg_list[j]['message_type'] == "80" or msg_list[j]['message_type'] == "81") and msg_list[j]['ip_dest']==msg_list[i]['ip_dest']:
                        break
                    if msg_list[j]['message_type']  == "00":
                        find_request = 1
                        break
            if find_request == 0:
                del_request_count=del_request_count+1
                msg_list[i]['label']=2


#写
    for i in range(len(msg_list)):
        xlswrite(worksheet, i+2, msg_list[i])

    workbook.close()
#########################
    #每个特征维度one-hot编码
    mac_dest_encode=torch.from_numpy(one_hot_encode(msg_list,'mac_dest'))
    mac_src_encode=torch.from_numpy(one_hot_encode(msg_list,'mac_src'))
    protocol_encode=torch.from_numpy(one_hot_encode(msg_list,'protocol'))
    port_src_encode=torch.from_numpy(one_hot_encode(msg_list,'port_src'))
    port_dest_encode=torch.from_numpy(one_hot_encode(msg_list,'port_dest'))
    client_id_encode=torch.from_numpy(one_hot_encode(msg_list,'client_id'))
    session_id_encode=torch.from_numpy(one_hot_encode(msg_list,'session_id'))
    protocol_version_encode=torch.from_numpy(one_hot_encode(msg_list,'protocol_version'))
    ip_src_encode=torch.from_numpy(one_hot_encode(msg_list,'ip_src'))
    ip_dest_encode=torch.from_numpy(one_hot_encode(msg_list,'ip_dest'))
    service_id_encode=torch.from_numpy(one_hot_encode(msg_list,'service_id'))
    method_id_encode=torch.from_numpy(one_hot_encode(msg_list,'method_id'))
    interface_version_encode=torch.from_numpy(one_hot_encode(msg_list,'interface_version'))
    message_type_encode=torch.from_numpy(one_hot_encode(msg_list,'message_type'))
    return_code_encode=torch.from_numpy(one_hot_encode(msg_list,'return_code'))
    label_encode=torch.from_numpy(one_hot_encode(msg_list,'label'))



#拼接各个特征维度的one-hot编码
    features=[]   #one-hot encode features
    for i in range(len(msg_list)):
        tmp_list=torch.cat([mac_dest_encode[i],mac_src_encode[i],protocol_encode[i],ip_src_encode[i],ip_dest_encode[i],port_dest_encode[i],port_src_encode[i],service_id_encode[i],method_id_encode[i],client_id_encode[i],session_id_encode[i],protocol_version_encode[i],interface_version_encode[i],message_type_encode[i],return_code_encode[i],label_encode[i]])

        features.append(list(tmp_list.numpy()))

    features=torch.tensor(features)
    features_input = features.shape[1]-3
    print("特征维度+标签的独热编码的shape: ",features.shape)

    list_index=[]
    seq_size=20;
    X=[]    #输入特征（样本数，seq包，特征195）
    Y=[]     #标签(样本数，seq包，标签3)
    for i in range(len(msg_list)):
        if i not in list_index:
            list_index.append(i)
            seq_index_tmp=0
            label=0
            features_seq=[[0]*features_input for row in range(seq_size)]
            features_seq[seq_index_tmp] = list(features[i][0:features_input].numpy())
            if list(features[i][features_input:features.shape[1]].numpy()).index(1) != 0:
                label = list(features[i][features_input:features.shape[1]].numpy()).index(1)
            seq_index_tmp=seq_index_tmp+1
            for j in range(i+1,len(msg_list)):
                if j not in list_index:
                    if is_sequence(msg_list[i],msg_list[j]):
                        features_seq[seq_index_tmp] = list(features[j][0:features_input].numpy())
                        if list(features[j][features_input:features.shape[1]].numpy()).index(1) !=0:
                            label=list(features[j][features_input:features.shape[1]].numpy()).index(1)
                        seq_index_tmp = seq_index_tmp + 1
                        list_index.append(j)
            X.append(list(features_seq))
            Y.append(label)

    X=torch.tensor(X,dtype=torch.float32)
    Y=torch.tensor(Y,dtype=torch.float32).reshape(1,-1)

    return X,Y,features_input,del_response_count,del_request_count

X1,Y1,features_input1,del_response_count1,del_request_count1 = data_handle('../data/features_1.xlsx','../traces/trace_1.pcap')
X2,Y2,features_input2,del_response_count2,del_request_count2 = data_handle('../data/features_2.xlsx','../traces/trace_2.pcap')
X3,Y3,features_input3,del_response_count3,del_request_count3 = data_handle('../data/features_3.xlsx','../traces/trace_3.pcap')
X4,Y4,features_input4,del_response_count4,del_request_count4 = data_handle('../data/features_4.xlsx','../traces/trace_4.pcap')
X5,Y5,features_input5,del_response_count5,del_request_count5 = data_handle('../data/features_5.xlsx','../traces/trace_5.pcap')
X6,Y6,features_input6,del_response_count6,del_request_count6 = data_handle('../data/features_6.xlsx','../traces/trace_6.pcap')
X7,Y7,features_input7,del_response_count7,del_request_count7 = data_handle('../data/features_7.xlsx','../traces/trace_7.pcap')
X8,Y8,features_input8,del_response_count8,del_request_count8 = data_handle('../data/features_8.xlsx','../traces/trace_8.pcap')

X_train = torch.cat((X1,X2,X3))
Y_train = torch.cat((Y1,Y2,Y3)).reshape(1,-1)
X_test=X4
Y_test=Y4
X_test_2=X5
Y_test_2=Y5
X_test_3=X6
Y_test_3=Y6
X_test_4=X7
Y_test_4=Y7
X_test_5=X8
Y_test_5=Y8


del_response_count_train=del_response_count1+del_response_count2+del_response_count3
del_request_count_train=del_request_count1+del_request_count2+del_request_count3
print("X_train:",X_train.shape)
print("Y_train: ",Y_train.shape)
print("del_response_attack: ", del_response_count_train)
print("del_request_attack: ", del_request_count_train)
print("X_test:",X_test.shape)
print("Y_test: ",Y_test.shape)
print("X_test_2:",X_test_2.shape)
print("Y_test_2: ",Y_test_2.shape)
print("X_test_3:",X_test_3.shape)
print("Y_test_3: ",Y_test_3.shape)
print("X_test_4:",X_test_4.shape)
print("Y_test_4: ",Y_test_4.shape)
print("X_test_5:",X_test_5.shape)
print("Y_test_5: ",Y_test_5.shape)

seq_num=X_train.shape[0]

torch.save(X_train,'../data/train_X.pkl')
torch.save(Y_train,'../data/train_Y.pkl')
torch.save(X_test,'../data/test_X.pkl')
torch.save(Y_test,'../data/test_Y.pkl')
torch.save(X_test_2,'../data/test_X_2.pkl')
torch.save(Y_test_2,'../data/test_Y_2.pkl')
torch.save(X_test_3,'../data/test_X_3.pkl')
torch.save(Y_test_3,'../data/test_Y_3.pkl')
torch.save(X_test_4,'../data/test_X_4.pkl')
torch.save(Y_test_4,'../data/test_Y_4.pkl')
torch.save(X_test_5,'../data/test_X_5.pkl')
torch.save(Y_test_5,'../data/test_Y_5.pkl')
class_weight=torch.FloatTensor([del_request_count_train/(len(X_train)-del_response_count_train-del_request_count_train),del_request_count_train/del_response_count_train,del_request_count_train/del_request_count_train])
print("class_weight",class_weight)
