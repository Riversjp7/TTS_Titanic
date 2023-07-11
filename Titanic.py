import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn import preprocessing
import numpy as np

def teacher(X_train, y_train,X_test,y_test,model):
    model.fit(X_train,y_train)
    modelP=model.predict(X_test)
    modelXor=np.sum(modelP^y_test)
    perc= 1-(modelXor/len(y_test))
    return modelXor,perc

train=pd.read_csv("/Users/Launch/Coding Projects/TTS final project/clean_titanic.csv", 
            usecols=["Pclass","Sex","Age","SibSp","Parch","Survived"])

X=train.loc[:, train.columns != 'Survived']
y=train["Survived"]
X.loc[:,"Sex"]=X.Sex.replace({'male':1,'female':0})
X=preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X,y ,
                                   random_state=0,  
                                   shuffle=True)

svm=SVC(kernel='linear',C=1.0, random_state=0)
knn= KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
tree_mod= DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=0)
lr=LogisticRegression()
gnb=GaussianNB()
#rf=RandomForestClassifier(max_depth=4,n_estimators=300)

model_names=['svm','knn','tree','lr','gnb']
model_tests=[svm,knn,tree_mod,lr,gnb]

ensemble=VotingClassifier(estimators=list(zip(model_names,model_tests)),voting='hard')

all_wrong=[]
all_perc_right=[]
for m in model_tests:
    wrong,perc_right=teacher(X_train, y_train,X_test,y_test,m)
    all_wrong.append(wrong)
    all_perc_right.append(perc_right)
results=list(zip(model_names,all_wrong,all_perc_right))


ewrong,ep=teacher(X_train, y_train,X_test,y_test,ensemble)


e2wrong,e2p=teacher(X_train, y_train,X_test,y_test,ensemble.set_params(svm='drop',gnb='drop'))


all_wrong=np.append(all_wrong,[ewrong,e2wrong])
all_perc_right=np.append(all_perc_right,[ep,e2p,np.mean(all_perc_right)])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
l = np.arange(2)
w = np.arange(4)
mw, ml = np.meshgrid(w, l)
x, y = ml.ravel(), mw.ravel()

names=["SVM","KNN","Decision Tree","Log-Reg","GNB","All","Ensemble","Mean"]
label=list(map(lambda x: str(x[0])+" "+ str(x[1])+"%",zip(names,np.round(all_perc_right*100,1))))


N = 8
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = all_perc_right*100-75
width = np.pi / 8 * all_perc_right
colors = plt.cm.hsv(radii / 8.)

ax = plt.subplot(projection='polar')
ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5,label=label)
#ax.set_label=label
plt.legend(bbox_to_anchor=(0, 1.15))
ax.set_yticklabels(np.arange(76,82))
ax.set_xticklabels([])
plt.title("Accuracy Percentage by Model \n")
plt.show()
