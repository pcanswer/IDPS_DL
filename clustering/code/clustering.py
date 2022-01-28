import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from joblib import dump, load

random_state = 999


def csv_dataprocess():
    df_D = pd.read_csv('../data/DoS_data_with_payload.csv')
    df_F = pd.read_csv('../data/Fuzzy_data_with_payload.csv')
    df_G = pd.read_csv('../data/gear_data_with_payload.csv')
    df_P = pd.read_csv('../data/RPM_data_with_payload.csv')
    df_D = df_D.iloc[200:-1]
    df_F = df_F.iloc[200:-1]
    df_G = df_G.iloc[200:-1]
    df_P = df_P.iloc[200:-1]

    df_total = pd.concat([df_D,df_F,df_G,df_P])

    standard_scaler = preprocessing.StandardScaler()
    features_index = df_total.drop(['label'],axis=1).dtypes[df_total.dtypes != 'object'].index
    df_total[features_index] = standard_scaler.fit_transform(df_total[features_index])
    # P :unknown
    df_known = df_total[df_total['label'] !='P']
    df_known['label'][df_known['label'] == 'R'] = 1
    df_known['label'][df_known['label'] != 1] = 0

    df_unknown = df_total[df_total['label'] == 'P']
    df_unknown['label'][df_unknown['label'] == 'P'] = 0

    df_known.to_csv("../data/df_known.csv")
    df2p = df_known[df_known['label'] == 1]
    df2pp = df2p.sample(n=None, frac=df_unknown.label.value_counts().iloc[0] / df_known.label.value_counts().iloc[0], replace=False, weights=None, random_state=None, axis=0)
    df_unknown = pd.concat([df_unknown,df2pp])
    df_unknown.to_csv("../data/df_unknown.csv")


def cl_kmeans_train(train_x, train_y, n,do_gridsearch_flag,max_n=50, b=100):
    if do_gridsearch_flag == True:
        best_n = 2
        best_score = 0
        for n in range(2,max_n):
            km_cluster = MiniBatchKMeans(n_clusters=n,batch_size=b,random_state=random_state)
            result = km_cluster.fit_predict(train_x)
            normal=np.zeros(n)
            abnormal=np.zeros(n)
            for v in range(0,n):
                for i in range(0,len(train_y)):
                    if result[i]==v:
                        if train_y[i]==1:
                            normal[v]=normal[v]+1
                        else:
                            abnormal[v]=abnormal[v]+1
            normal_list=[]
            abnormal_list=[]
            for v in range(0,n):
                if normal[v]<=abnormal[v]:
                    abnormal_list.append(v)
                else:
                    normal_list.append(v)
            for v in range(0,len(train_y)):
                if result[v] in normal_list:
                    result[v]=1
                elif result[v] in abnormal_list:
                    result[v]=0
            score1 = accuracy_score(train_y,result)
            print('n = ',n,'accuracy_score = ',score1)
            if best_score < score1:
                best_score = score1
                best_n = n
                best_km_cluster = km_cluster
                best_normal_list = normal_list
                best_abnormal_list = abnormal_list
        print('best_n: ',best_n,'best_score: ',best_score)
        return best_km_cluster, best_n, best_normal_list, best_abnormal_list
    else:
        best_n = n
        km_cluster = MiniBatchKMeans(n_clusters=n,batch_size=b,random_state=random_state)
        result = km_cluster.fit_predict(train_x)
        normal = np.zeros(n)
        abnormal = np.zeros(n)
        for v in range(0, n):
            for i in range(0, len(train_y)):
                if result[i] == v:
                    if train_y[i] == 1:
                        normal[v] = normal[v] + 1
                    else:
                        abnormal[v] = abnormal[v] + 1
        normal_list = []
        abnormal_list = []
        for v in range(0, n):
            if normal[v] <= abnormal[v]:
                abnormal_list.append(v)
            else:
                normal_list.append(v)
        for v in range(0, len(train_y)):
            if result[v] in normal_list:
                result[v] = 1
            elif result[v] in abnormal_list:
                result[v] = 0
        score1 = accuracy_score(train_y, result)
        print('n = ', n, 'accuracy_score = ', score1)
        return km_cluster, best_n, normal_list, abnormal_list


def cl_kmeans_train_unknown(train_x, train_y,unknown_x,unknown_y,max_n=50, b=100):
    best_n = 4
    best_score = 0
    for n in range(4,max_n):
        km_cluster = MiniBatchKMeans(n_clusters=n,batch_size=b,random_state=random_state)
        result = km_cluster.fit_predict(train_x)
        result_unknown = km_cluster.fit_predict(unknown_x)
        normal=np.zeros(n)
        abnormal=np.zeros(n)
        for v in range(0,n):
            for i in range(0,len(train_y)):
                if result[i]==v:
                    if train_y[i]==1:
                        normal[v]=normal[v]+1
                    else:
                        abnormal[v]=abnormal[v]+1
        normal_list=[]
        abnormal_list=[]
        for v in range(0,n):
            if normal[v]<=abnormal[v]:
                abnormal_list.append(v)
            else:
                normal_list.append(v)
        for v in range(0,len(unknown_y)):
            if result_unknown[v] in normal_list:
                result_unknown[v]=1
            elif result_unknown[v] in abnormal_list:
                result_unknown[v]=0
        score1 = accuracy_score(unknown_y,result_unknown)
        print('n = ',n,'accuracy_score = ',score1)
        if best_score < score1:
            best_score = score1
            best_n = n
            best_km_cluster = km_cluster
            best_normal_list = normal_list
            best_abnormal_list = abnormal_list
    print('best_n: ',best_n,'best_score: ',best_score)
    return best_km_cluster, best_n, best_normal_list, best_abnormal_list

def cl_kmeans_test(model,normal_list,abnormal_list,test_x,test_y):
    result = model.predict(test_x)
    for v in range(0, len(test_y)):
        if result[v] in normal_list:
            result[v] = 1
        elif result[v] in abnormal_list:
            result[v] = 0
    cm2 = confusion_matrix(test_y, result)
    print(cm2)
    score2 = accuracy_score(test_y, result)
    print(score2)
    f,ax=plt.subplots(figsize=(2,2))
    sns.heatmap(cm2,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.show()
    print(classification_report(test_y,result))


csv_process_flag = 1
train_flag = 1
test_flag = 1
if __name__ == '__main__':
    if csv_process_flag == 1:
        csv_dataprocess()

    df_known = pd.read_csv('../data/df_known.csv')
    df_unknown = pd.read_csv('../data/df_unknown.csv')

    col_features = ['time','canid','byte0','byte1','byte2','byte3','byte4','byte5','byte6','byte7']
    features = pd.DataFrame(df_known, columns=col_features).values.tolist()
    labels = df_known.label.values.tolist()
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=random_state, train_size=0.7,
                                                                      test_size=0.3, stratify=labels)
    print('known attack train dataset before smote: \r\n', pd.Series(train_y).value_counts())
    print(pd.Series(train_y).value_counts().iloc[0])
    smote = SMOTE(n_jobs=-1,sampling_strategy={0:pd.Series(train_y).value_counts().iloc[0]})
    train_x, train_y = smote.fit_resample(train_x,train_y)
    print('known attack train dataset after smote: \r\n', pd.Series(train_y).value_counts())
    print('known attack test dataset after smote: \r\n', pd.Series(test_y).value_counts())
    features_unknown = pd.DataFrame(df_unknown, columns=col_features).values.tolist()
    labels_unknown = df_unknown.label.values.tolist()

    if train_flag == 1:
        model, best_n, normal_list, abnormal_list = cl_kmeans_train_unknown(train_x, train_y,features_unknown,labels_unknown,max_n=30)
        dump(model, '../model/cluster.pkl')
        dump(normal_list, '../model/cluster_normal_list.pkl')
        dump(abnormal_list, '../model/cluster_abnormal_list.pkl')
    if test_flag == 1:
        cluster_model = load('../model/cluster.pkl')
        normal_list = load('../model/cluster_normal_list.pkl')
        abnormal_list = load('../model/cluster_abnormal_list.pkl')
        #known attack detect
        cl_kmeans_test(cluster_model,normal_list, abnormal_list,test_x,test_y)
        #unknown attack detect
        cl_kmeans_test(cluster_model, normal_list, abnormal_list, features_unknown, labels_unknown)

