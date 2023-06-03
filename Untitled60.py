#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder


# In[41]:


df = pd.read_csv("E:/vachan/heart_Disease.csv")


# In[42]:


df.info()


# In[43]:


df.columns


# In[44]:


df.shape


# In[46]:


df.isnull().sum()


# In[48]:


#target - have disease or not (1=yes, 0=no)
df.target.value_counts()


# In[49]:


#Bar plot 
sns.countplot(x='target', data = df, palette='hot')
plt.title("Bar plot of target")


# In[50]:


#age - age in years
df.age.value_counts()


# In[51]:


#Histogram
plt.hist(df.age, bins = 'auto', facecolor = 'red')
plt.xlabel('age')
plt.ylabel('counts')
plt.title('Histogram of age')


# In[52]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["age"].plot.box(color=props2, patch_artist = True, vert= False)


# In[53]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('target~age', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[54]:


# 0.000075 ie. p_value is lessthan 0.05 Ho is reject; Good predictor


# In[55]:


#sex - (1 = male; 0 = female)
df.sex.value_counts()


# In[56]:


#Bar plot 
sns.countplot(x='sex', data = df, palette='hot')
plt.title("Bar plot of sex")


# In[168]:


# group the data by sex and calculate the avererage disease by each  gender type
df_sex_type = df.groupby('sex')['target'].mean()
print(df_sex_type)


# In[169]:


# visualize the df_customer_type
df_sex_type.plot(kind='bar', title ='Average disease occurs based on sex type')


# In[57]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["sex"].plot.box(color=props2, patch_artist = True, vert= False)


# In[61]:


# hypothesis test
from scipy.stats import chi2_contingency
df_sex=pd.crosstab(df.target, df.sex)
chi2_contingency(df_sex, correction = False)


# In[62]:


#1.0071642033238865e-06 ie. p_value is lessthan 0.05 Ho is reject; Good predictor


# In[63]:


#cp - chest pain type
df.cp.value_counts()


# In[64]:


#Bar plot 
sns.countplot(x='cp', data = df, palette='hot')
plt.title("Bar plot of cp")


# In[65]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["cp"].plot.box(color=props2, patch_artist = True, vert= False)


# In[66]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('target~cp', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[67]:


#2.469712e-15  ie. p_value is lessthan 0.05 Ho is reject; Good predictor


# In[68]:


#trestbps - resting blood pressure (in mm Hg on admission to the hospital)
df.trestbps.value_counts()


# In[69]:


#Histogram
plt.hist(df.trestbps, bins = 'auto', facecolor = 'red')
plt.xlabel('trestbps')
plt.ylabel('counts')
plt.title('Histogram of trestbps')


# In[70]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["trestbps"].plot.box(color=props2, patch_artist = True, vert= False)


# In[71]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('target~trestbps', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[72]:


#0.011546 ie. p_value is greaterthan 0.05 Ho is accept; bad predictor


# In[73]:


#chol - serum cholestoral in mg/dl
df.chol.value_counts()


# In[74]:


#Histogram
plt.hist(df.chol, bins = 'auto', facecolor = 'red')
plt.xlabel('chol')
plt.ylabel('counts')
plt.title('Histogram of chol')


# In[76]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["chol"].plot.box(color=props2, patch_artist = True, vert= False)


# In[77]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('target~chol', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[78]:


#0.13879 ie. p_value is greaterthan 0.05 Ho is accept; bad predictor


# In[79]:


#fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
df.fbs.value_counts()


# In[80]:


#Bar plot 
sns.countplot(x='fbs', data = df, palette='hot')
plt.title("Bar plot of fbs")


# In[81]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["fbs"].plot.box(color=props2, patch_artist = True, vert= False)


# In[82]:


# hypothesis test
from scipy.stats import chi2_contingency
df_sex=pd.crosstab(df.target, df.fbs)
chi2_contingency(df_sex, correction = False)


# In[83]:


#0.625 ie. p_value is greaterthan 0.05 Ho is accept; bad predictor


# In[84]:


#restecg - resting electrocardiographic results
df.restecg.value_counts()


# In[85]:


#Bar plot 
sns.countplot(x='restecg', data = df, palette='hot')
plt.title("Bar plot of fbs")


# In[86]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["restecg"].plot.box(color=props2, patch_artist = True, vert= False)


# In[87]:


# hypothesis test
from scipy.stats import chi2_contingency
df_sex=pd.crosstab(df.target, df.restecg)
chi2_contingency(df_sex, correction = False)


# In[88]:


#0.0066 ie. p_value is lessthan 0.05 Ho is reject; good predictor


# In[89]:


#thalach - maximum heart rate achieved
df.thalach.value_counts()


# In[90]:


#Histogram
plt.hist(df.thalach, bins = 'auto', facecolor = 'red')
plt.xlabel('thalach')
plt.ylabel('counts')
plt.title('Histogram of thalach')


# In[91]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["thalach"].plot.box(color=props2, patch_artist = True, vert= False)


# In[92]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('target~thalach', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[93]:


#1.697338e-14 ie. p_value is lessthan 0.05 Ho is reject; good predictor


# In[95]:


#exang - exercise induced angina (1 = yes; 0 = no)
df.exang.value_counts()


# In[96]:


#Bar plot 
sns.countplot(x='exang', data = df, palette='hot')
plt.title("Bar plot of exang")


# In[97]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["exang"].plot.box(color=props2, patch_artist = True, vert= False)


# In[98]:


# hypothesis test
from scipy.stats import chi2_contingency
df_sex=pd.crosstab(df.target, df.exang)
chi2_contingency(df_sex, correction = False)


# In[99]:


#2.9027370724511966e-14 ie. p_value is lessthan 0.05 Ho is reject; good predictor


# In[100]:


#oldpeak - ST depression induced by exercise relative to rest
df.oldpeak.value_counts()


# In[101]:


#Histogram
plt.hist(df.oldpeak, bins = 'auto', facecolor = 'red')
plt.xlabel('oldpeak')
plt.ylabel('counts')
plt.title('Histogram of oldpeak')


# In[102]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["oldpeak"].plot.box(color=props2, patch_artist = True, vert= False)


# In[103]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('target~oldpeak', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[104]:


#4.085346e-15 ie. p_value is lessthan 0.05 Ho is reject; good predictor


# In[105]:


#slope - the slope of the peak exercise ST segment
df.slope.value_counts()


# In[106]:


#Bar plot 
sns.countplot(x='slope', data = df, palette='hot')
plt.title("Bar plot of slope")


# In[107]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["slope"].plot.box(color=props2, patch_artist = True, vert= False)


# In[108]:


# hypothesis test
from scipy.stats import chi2_contingency
df_sex=pd.crosstab(df.target, df.slope)
chi2_contingency(df_sex, correction = False)


# In[109]:


#4.8306819342768186e-11 ie. p_value is lessthan 0.05 Ho is reject; good predictor


# In[110]:


#ca - number of major vessels (0-3) colored by flourosopy
df.ca.value_counts()


# In[111]:


#Bar plot 
sns.countplot(x='ca', data = df, palette='hot')
plt.title("Bar plot of ca")


# In[112]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["ca"].plot.box(color=props2, patch_artist = True, vert= False)


# In[113]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('target~ca', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[114]:


#1.491539e-12 ie. p_value is lessthan 0.05 Ho is reject; good predictor 


# In[115]:


#thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
df.thal.value_counts()


# In[116]:


#Bar plot 
sns.countplot(x='thal', data = df, palette='hot')
plt.title("Bar plot of thal")


# In[117]:


#boxplot
props2= dict(boxes= 'red', whiskers='green', medians ='black', caps='blue')
df["thal"].plot.box(color=props2, patch_artist = True, vert= False)


# In[118]:


# hypothesis test
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('target~thal', data =df).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)


# In[119]:


#7.624880e-10 ie. p_value is lessthan 0.05 Ho is reject; good predictor


# In[121]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
x = df.iloc[:,[0,1,2,6,7,8,9,10,11,12]]
vif_data=pd.DataFrame()
vif_data['feature']=x.columns
vif_data['VIF']=[variance_inflation_factor(x.values, i)
                         for i in range(len(x.columns))]
print(vif_data)


# In[123]:


df=df.drop(['thal', 'trestbps', 'thalach','age', 'chol', 'fbs'], axis=1)


# In[126]:


#Model
X=df.loc[:,df.columns!= 'target']
y=df.loc[:,df.columns== 'target']
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression(solver='liblinear', random_state=0)
model1.fit(X,y)
model1.intercept_


# In[127]:


model1.coef_


# In[129]:


#Predictions
y_pred = model1.predict(X)


# In[130]:


#Confusion Matrix
from sklearn import metrics
cm=metrics.confusion_matrix(y, y_pred)
print(cm)


# In[131]:


model1.score(X,y)


# In[132]:


# classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))


# In[134]:


# ROC - Reciever Operating Characterstic curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
y_pred_prob = model1.predict_proba(X)
fpr,tpr, thresholds = roc_curve(df['target'], y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)


# In[135]:


# ROC Curve
plt.title('ROC Curve For LogReg: liblinear')
plt.xlabel('False Positive Rate (1-specificity)')
plt.ylabel('True positive Rate (sensitivity)')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(fpr, tpr, label = 'AUC='+str(roc_auc))
plt.legend(loc=4)
plt.show()


# In[137]:


# SMOTE
disno = df[df.target == 0]
dis = df[df.target == 1]


# In[138]:


from sklearn.utils import resample
dis_oversample = resample(dis,
                          replace=True,
                          n_samples=len(disno),
                          random_state=27)


# In[140]:


dfsmote = pd.concat([disno, dis_oversample])
dfsmote.target.value_counts()


# In[141]:


X2 = dfsmote.loc[:, dfsmote.columns != 'target']
y2 = dfsmote.loc[:, dfsmote.columns == 'target']


# In[142]:


from sklearn.linear_model import LogisticRegression
model2=LogisticRegression(solver='liblinear', random_state=0)
model2.fit(X2,y2)
model2.intercept_


# In[143]:


model2.coef_


# In[144]:


#Predictions
y_pred = model2.predict(X2)


# In[145]:


#Confusion Matrix
from sklearn import metrics
cm=metrics.confusion_matrix(y2, y_pred)
print(cm)


# In[146]:


model2.score(X2,y2)


# In[147]:


# classification report
from sklearn.metrics import classification_report
print(classification_report(y2, y_pred))


# In[149]:


# ROC - Reciever Operating Characterstic curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
y_pred_prob2 = model2.predict_proba(X2)
fpr2,tpr2, thresholds2 = roc_curve(dfsmote['target'], y_pred_prob[:, 1])
roc_auc2 = auc(fpr2, tpr2)
print(roc_auc2)


# In[150]:


# ROC Curve
plt.title('ROC Curve For LogReg: liblinear SMOTE')
plt.xlabel('False Positive Rate (1-specificity)')
plt.ylabel('True positive Rate (sensitivity)')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(fpr2, tpr2, label = 'AUC='+str(roc_auc2))
plt.legend(loc=4)
plt.show()


# In[152]:


# GLM Method
import statsmodels.api as sm
import statsmodels.formula.api as smf
model3 = smf.glm(formula= 'target~sex+cp+restecg+exang+oldpeak+ca', data=df, family=sm.families.Binomial())
result3 = model3.fit()
print(result3.summary())


# In[155]:


predictions3=result3.predict()
predictions_nominal3 = [0 if x < 0.5 else 1 for x in predictions3]
predictions_nominal3


# In[156]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df['target'], predictions_nominal3))


# In[157]:


# ROC - Reciever Operating Characterstic curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr3,tpr3, thresholds3 = roc_curve(df['target'], predictions3)
roc_auc3 = auc(fpr3, tpr3)
print(roc_auc3)


# In[158]:


# ROC Curve
plt.title('ROC Curve For LogReg: liblinear')
plt.xlabel('False Positive Rate (1-specificity)')
plt.ylabel('True positive Rate (sensitivity)')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(fpr3, tpr3, label = 'AUC='+str(roc_auc3))
plt.legend(loc=4)
plt.show()


# In[159]:


# Classification Report
print(classification_report(df['target'], predictions_nominal3, digits=3))


# In[160]:


# GLM Method
import statsmodels.api as sm
import statsmodels.formula.api as smf
model4 = smf.glm(formula= 'target~sex+cp+exang+oldpeak+ca', data=df, family=sm.families.Binomial())
result4 = model4.fit()
print(result4.summary())


# In[161]:


predictions4=result4.predict()
predictions_nominal4 = [0 if x < 0.5 else 1 for x in predictions4]
predictions_nominal4


# In[164]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df['target'], predictions_nominal4))


# In[165]:


# ROC - Reciever Operating Characterstic curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr4,tpr4, thresholds4 = roc_curve(df['target'], predictions4)
roc_auc4 = auc(fpr4, tpr4)
print(roc_auc4)


# In[166]:


# ROC Curve
plt.title('ROC Curve For LogReg: liblinear')
plt.xlabel('False Positive Rate (1-specificity)')
plt.ylabel('True positive Rate (sensitivity)')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot(fpr4, tpr4, label = 'AUC='+str(roc_auc4))
plt.legend(loc=4)
plt.show()


# In[167]:


# Classification Report
print(classification_report(df['target'], predictions_nominal4, digits=3))


# In[ ]:




