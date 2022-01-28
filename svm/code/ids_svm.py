from cgi import test
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


train_flag = 0
validation_flag = 1
test_flag = 1

if __name__ == '__main__':
    df = pd.read_csv('../data/ADAS_blffuz_2.csv')
    print(df.label.value_counts())
    df = df.drop(df.index[0:200])
    df.time = df.time * 10

    col_features = ['time','canid','byte0','byte1','byte2','byte3','byte4','byte5','byte6','byte7','label']
    features = pd.DataFrame(df,columns=col_features)
    labels = df.label
    # scaler = StandardScaler()
    # train_data,test_data,train_label,test_label=train_test_split(features,labels,random_state=1,train_size=0.8,test_size=0.2,stratify=labels)
    train_data,test_val_data,train_label,test_val_label=train_test_split(features,labels,random_state=1,train_size=0.7,test_size=0.3,stratify=labels)
   

    test_data,val_data,test_label,val_label=train_test_split(test_val_data,test_val_label,random_state=1,train_size=0.5,test_size=0.5,stratify=test_val_label)

    if train_flag == 1:
        model = svm.SVC(C=1.1, kernel='rbf', gamma=0.15, decision_function_shape='ovo', max_iter=500)
        model.fit(train_data,train_label.ravel())
        dump(model, '../model/svm.pkl')
        train_score = model.score(train_data,train_label)
        print("训练集：",train_score)
    

    if validation_flag ==1:
        model = load('../model/svm.pkl')
        val_predict = model.predict(val_data)
        val_true=val_label
        val_score = model.score(val_data,val_label)
        print(confusion_matrix(val_predict,val_true))
        print("验证集：",val_score)

        precision,recall,fscore,none= precision_recall_fscore_support(val_true, val_predict, average='weighted')
        print('Precision of svm: '+(str(precision)))
        print('Recall of svm: '+(str(recall)))
        print('F1-score of svm: '+(str(fscore)))
        print(classification_report(val_true,val_predict))
        cm=confusion_matrix(val_true,val_predict)
        f,ax=plt.subplots(figsize=(2,2))
        sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        plt.show()

    if test_flag == 1:
        model = load('../model/svm.pkl')
        test_predict = model.predict(test_data)
        test_true=test_label
        test_score = model.score(test_data,test_label)
        print(confusion_matrix(test_predict,test_true))
        print("测试集：",test_score)


        precision,recall,fscore,none= precision_recall_fscore_support(test_true, test_predict, average='weighted')
        print('Precision of svm: '+(str(precision)))
        print('Recall of svm: '+(str(recall)))
        print('F1-score of svm: '+(str(fscore)))
        print(classification_report(test_true,test_predict))
        cm=confusion_matrix(test_true,test_predict)
        f,ax=plt.subplots(figsize=(2,2))
        sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        plt.show()

    if validation_flag ==1:
        model = load('../model/svm.pkl')
        val_predict = model.predict(val_data)
        val_true=val_label
        val_score = model.score(val_data,val_label)
        print(confusion_matrix(val_predict,val_true))
        print("验证集：",val_score)

        precision,recall,fscore,none= precision_recall_fscore_support(val_true, val_predict, average='weighted')
        print('Precision of svm: '+(str(precision)))
        print('Recall of svm: '+(str(recall)))
        print('F1-score of svm: '+(str(fscore)))
        print(classification_report(val_true,val_predict))
        cm=confusion_matrix(val_true,val_predict)
        f,ax=plt.subplots(figsize=(2,2))
        sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        plt.show()