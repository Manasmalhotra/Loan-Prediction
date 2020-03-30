import pandas as pd
import numpy as np
import pandas_ml as pdml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.utils import resample

df=pd.read_csv('Finalized.csv')

df_majority = df[df['Loan_Status_Y']==1]
df_minority = df[df['Loan_Status_Y']==0]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=309,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
data= pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts

print(data['Loan_Status_Y'].value_counts())
x=data.drop(['Loan_Status_Y','Loan_ID'],axis=1)
y=data['Loan_Status_Y']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
p=g=m=0
r=RandomForestClassifier(n_estimators=10,criterion='gini',random_state=10,max_depth=8,min_samples_leaf=3)
r.fit(x_train,y_train)
print(r.score(x_test,y_test))
print(r.feature_importances_.round(2))
l=LogisticRegression(solver='lbfgs',max_iter=600)
l.fit(x_train,y_train)
print(l.score(x_test,y_test))

e=pd.read_csv('results.csv')
s=pd.read_csv('test.csv')
e['Loan_ID']=s['Loan_ID']
s.drop(['Loan_ID'],axis=1,inplace=True)
#s.drop(['Unnamed: 0'],axis=1,inplace=True)
s.to_csv('tett.csv')
e['Loan_Status']=l.predict(s)
e['Loan_Status'].replace(to_replace=0,value='N',inplace=True)
e['Loan_Status'].replace(to_replace=1,value='Y',inplace=True)
e.to_csv('Submission3.csv',index=False)

