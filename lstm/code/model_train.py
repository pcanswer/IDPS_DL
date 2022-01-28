import torch
from torch import nn


X_train=torch.load('../data/train_X.pkl')
Y_train=torch.load('../data/train_Y.pkl')
X_test=torch.load('../data/test_X.pkl')
Y_test=torch.load('../data/test_Y.pkl')
X_test_2=torch.load('../data/test_X_2.pkl')
Y_test_2=torch.load('../data/test_Y_2.pkl')
X_test_3=torch.load('../data/test_X_3.pkl')
Y_test_3=torch.load('../data/test_Y_3.pkl')
X_test_4=torch.load('../data/test_X_4.pkl')
Y_test_4=torch.load('../data/test_Y_4.pkl')
X_test_5=torch.load('../data/test_X_5.pkl')
Y_test_5=torch.load('../data/test_Y_5.pkl')
features_input=193
seq_num=X_train.shape[0]

#RNN
EPOCH=300
BATCH_SIZE=300
LR=0.05
class_weight=torch.tensor([0.0356, 0.8043, 1.0000])

tt_factor=1
hidden_dim = 20
layer = 1
n_class = 3
drop_out = 0.2


class LSTMnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMnet, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out

model = LSTMnet(features_input,hidden_dim,layer,n_class)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(weight=class_weight)


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
    sum = [dic_result['0->0'] + dic_result['0->1'] + dic_result['0->2'],
           dic_result['1->0'] + dic_result['1->1'] + dic_result['1->2'],
           dic_result['2->0'] + dic_result['2->1'] + dic_result['2->2']]
    matrices=torch.tensor([[dic_result['0->0']/(sum[0]),dic_result['0->1']/(sum[0]),dic_result['0->2']/(sum[0])],[dic_result['1->0']/(sum[1]),dic_result['1->1']/(sum[1]),dic_result['1->2']/(sum[1])],[dic_result['2->0']/(sum[2]),dic_result['2->1']/(sum[2]),dic_result['2->2']/(sum[2])]])
    print(matrices)



if __name__ =='__main__':
    ##train
    for epoch in range(EPOCH):
        for i in range(int(tt_factor * seq_num / BATCH_SIZE)):
            output = model(X_train[BATCH_SIZE * i:BATCH_SIZE * i + BATCH_SIZE])
            loss = criterion(output, Y_train[0][BATCH_SIZE * i:BATCH_SIZE * i + BATCH_SIZE].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch:{:<2d} | loss:{:<6.4f}'.format(epoch, loss))

    model.eval()
    torch.save(model, '../model/model.pkl')
    print(model.state_dict())

    print("train")
    test_output = model(X_train)
    predict_y = torch.max(test_output, 1)[1].numpy()
    correct = (predict_y == Y_train[0].numpy()).astype(int).sum()
    totoal = Y_train[0].size(0)
    accuracy = float(correct) / float(totoal)
    print('test_dataset: accuracy:{:<4.2f} | correct:{:<2d} | totoal:{:<2d}'.format(accuracy, correct, totoal))

    print("test_1")
    test(X_test, Y_test, model)
    print("test_2")
    test(X_test_2, Y_test_2, model)
    print("test_3")
    test(X_test_3, Y_test_3, model)
    print("test_4")
    test(X_test_4, Y_test_4, model)
    print("test_5")
    test(X_test_5, Y_test_5, model)