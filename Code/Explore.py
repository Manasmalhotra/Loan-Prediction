import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

pd.set_option('display.max_columns', None)
sns.set()

data=pd.read_csv('train_1.csv')
sns.boxplot(data['LoanAmount'])
plt.show()

w=data['ApplicantIncome']+data['CoapplicantIncome']
sns.boxplot(w)
plt.show()


data['Total']=data['CoapplicantIncome']+data['ApplicantIncome']
data.drop(['CoapplicantIncome','ApplicantIncome'],axis=1,inplace=True)

print(data.isnull().sum())

#############################Treat Outliers########################################################################################################################
data['LoanAmount']=np.log(data['LoanAmount'])
data['Total']=np.log(data['Total'])
data.loc[data['Dependents']=="3+",'Dependents']=3
#data=pd.get_dummies(data,columns=['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status'],drop_first=True)
################################Missing Values################################################################################################################3
data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)
data['Married'].fillna(data['Married'].mode()[0],inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median().round(2),inplace=True)
data=pd.get_dummies(data,columns=['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status'],drop_first=True)
    
print(data.isnull().sum())
w=data.drop(['Loan_ID'],axis=1)
imp = IterativeImputer(estimator=RandomForestClassifier(n_estimators=100),max_iter=100, random_state=0)
imp.fit(w)
w=pd.DataFrame(imp.transform(w))
data['Credit_History']=w[3]
print(data.isnull().sum())
#data['EMI']=data['LoanAmount']/(data['Loan_Amount_Term']/12)
#data.drop(['LoanAmount','Loan_Amount_Term'],axis=1,inplace=True)
data.to_csv('Finalized.csv',index=False)
