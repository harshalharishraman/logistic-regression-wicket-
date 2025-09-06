import matplotlib.pyplot as pmat
import seaborn as sea
import pandas as pd
import numpy as ny
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import *
from statsmodels.stats.outliers_influence import variance_inflation_factor as vf

lgr=LogisticRegression()
lr=LinearRegression()
lcodr=LabelEncoder()
tts=train_test_split

de=pd.read_csv(r'D:\py practice\deliveries.csv')
cv=list(de.columns)

for i in cv:
    if de[i].dtype=='object':
        if not de[i].mode().empty:
            de[i]=de[i].fillna(de[i].mode()[0])
            de[i]=lcodr.fit_transform(de[i])
        else:
            de[i]=de[i].fillna('uk')
            de[i]=lcodr.fit_transform(de[i])

       
        
    else:
        de[i]=de[i].fillna(de[i].mean())
    if i in ['over', 'ball', 'extra_runs', 'total_runs', 'inning']:
            q1=de[i].quantile(0.25)
            q3=de[i].quantile(0.75)
            iqr=q3-q1
            lf=q1-(1.5*iqr)
            uf=q3+(1.5*iqr)
            de=de[(de[i]>=lf) & (de[i]<=uf)]

     

#print(de.isnull().sum())
dl=de.drop('is_wicket',axis=1)
t=de['is_wicket']
a='xyz'
x=dl.copy()

while True:
    vfd=pd.DataFrame()
    vfd['features']=x.columns
    vfd['vif']=[vf(x.values,i) for i in range(len(x.columns))]
    #vfd['vif']=pd.to_numeric(vfd['vif'],errors='coerce')
    mvf=vfd['vif'].max()
    if mvf<5:
          break
    a=vfd.loc[vfd['vif']==vfd['vif'].max(),'features'].values[0]
    dl=dl.drop(a,axis=1)
    x=dl.copy()

lq,qp,la,akq=tts(dl,t,train_size=0.8,random_state=42)
lgr.fit(lq,la)
ansg=lgr.predict(qp)
def sig(n):
    return 1 / (1 + ny.exp(-n))
ans=lgr.predict(qp)    
print(dl.columns)
print(f'accuracy score{accuracy_score(ans,akq)}\ncofusion matrix:{confusion_matrix(ans,akq)}\nclassification report:\n{classification_report(ans,akq)}')
lgt=lgr.decision_function(qp)
slgt=ny.sort(lgt)
yval=sig(slgt)
mx,mn=ny.argmax(slgt),ny.argmin(slgt)
xx,xn=slgt[mx],slgt[mn]
yx,yn=sig(xx),sig(xn)


pmat.figure(figsize=(8,6))
pmat.plot(slgt,yval,color='red',label='sigmoid curve')
pmat.xlabel('logit')
pmat.ylabel('sig transformed logits')
pmat.title('logistic regression sigmoid curve')
pmat.scatter(xx,yx,color='red',label='max logit')
pmat.axhline(y=0.5,color='Blue',linestyle='--')
pmat.scatter(xn,yn,color='blue',label='max logit')
pmat.legend()
pmat.show()



