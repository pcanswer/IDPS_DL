import torch
from torch import nn
from model_train import LSTMnet, features_input,hidden_dim,layer,n_class,LR,class_weight
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def test(X,Y,model):
    dic_result={'0->0':0,'0->1':0,'0->2':0,'1->0':0,'1->1':0,'1->2':0,'2->0':0,'2->1':0,'2->2':0}
    test_output = model(X)
    predict_y = torch.max(test_output, 1)[1].numpy()
    for i in range(len(Y[0])):
        if Y[0][i] == 0:
            if predict_y[i] == 1:
                dic_result['0->1'] = dic_result['0->1']+1
            elif predict_y[i] == 2:
                dic_result['0->2'] = dic_result['0->2'] + 1
            elif predict_y[i] == 0:
                dic_result['0->0'] = dic_result['0->0'] + 1
        elif Y[0][i] == 1:
            if predict_y[i] == 1:
                dic_result['1->1'] = dic_result['1->1']+1
            elif predict_y[i] == 2:
                dic_result['1->2'] = dic_result['1->2'] + 1
            elif predict_y[i] == 0:
                dic_result['1->0'] = dic_result['1->0'] + 1
        elif Y[0][i] == 2:
            if predict_y[i] == 1:
                dic_result['2->1'] = dic_result['2->1']+1
            elif predict_y[i] == 2:
                dic_result['2->2'] = dic_result['2->2'] + 1
            elif predict_y[i] == 0:
                dic_result['2->0'] = dic_result['2->0'] + 1
    correct = (predict_y == Y[0].numpy()).astype(int).sum()
    totoal = Y[0].size(0)
    accuracy = float(correct) / float(totoal)
    print('test_dataset: accuracy:{:<4.2f} | correct:{:<2d} | totoal:{:<2d}'.format(accuracy,correct, totoal))
    print(dic_result)
    sum=[dic_result['0->0']+dic_result['0->1']+dic_result['0->2'],dic_result['1->0']+dic_result['1->1']+dic_result['1->2'],dic_result['2->0']+dic_result['2->1']+dic_result['2->2']]
    matrices=torch.tensor([[dic_result['0->0']/(sum[0]),dic_result['0->1']/(sum[0]),dic_result['0->2']/(sum[0])],[dic_result['1->0']/(sum[1]),dic_result['1->1']/(sum[1]),dic_result['1->2']/(sum[1])],[dic_result['2->0']/(sum[2]),dic_result['2->1']/(sum[2]),dic_result['2->2']/(sum[2])]])
    print(matrices)
    print(classification_report(Y[0], predict_y))
    confu_matrix = confusion_matrix(Y[0],predict_y)
    print(confu_matrix)
    # plt.matshow(confu_matrix)
    # plt.colorbar()
    # plt.show()
    f,ax=plt.subplots(figsize=(3,3))
    sns.heatmap(confu_matrix,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.show()


if __name__ =='__main__':
    model_load = torch.load('../model/model.pkl')
    X_test_1 = torch.load('../data/test_X.pkl')
    Y_test_1 = torch.load('../data/test_Y.pkl')
    X_test_2 = torch.load('../data/test_X_2.pkl')
    Y_test_2 = torch.load('../data/test_Y_2.pkl')
    X_test_3 = torch.load('../data/test_X_3.pkl')
    Y_test_3 = torch.load('../data/test_Y_3.pkl')
    X_test_4 = torch.load('../data/test_X_4.pkl')
    Y_test_4 = torch.load('../data/test_Y_4.pkl')
    X_test_5 = torch.load('../data/test_X_5.pkl')
    Y_test_5 = torch.load('../data/test_Y_5.pkl')
    # print("test_1")
    # test(X_test,Y_test,model_load)
    # print("test_2")
    # test(X_test_2,Y_test_2,model_load)
    # print("test_3")
    # test(X_test_3,Y_test_3,model_load)
    # print("test_4")
    # test(X_test_4,Y_test_4,model_load)
    # print("test_5")
    # test(X_test_5,Y_test_5,model_load)
    X_test = torch.cat((X_test_1,X_test_2,X_test_3,X_test_4,X_test_5),0)
    Y_test = torch.cat([Y_test_1, Y_test_2, Y_test_3, Y_test_4, Y_test_5],1)
    test(X_test, Y_test, model_load)
