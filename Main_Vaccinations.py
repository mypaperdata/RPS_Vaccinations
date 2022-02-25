"""
Created on Fri Dec 17 11:55:14 2021
"""
# Orhan Erdem
# 11-18-2021
# This Program uses normalized vaccinations data, from Python_Norm_1m sheet of Vaccinations file, 
# in which the data on August 13, 2021 will be normalized to 100. This algorithm creates 
# 1. figures and output file with synthetic control and regression line. 
#2. placebo analysis with placebo counties. 
# All files must be in the same file folder.

# Import additional modules
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
from datetime import datetime
from typing import List

#Import the data and calculate the normalization coefficients
wb0 = pd.read_excel('Vaccinations.xlsx',sheet_name='Python')  #This is the raw vaccinations data. 
wb = pd.read_excel('Vaccinations.xlsx',sheet_name='Python_Norm_1m')    #This is the normalized data + placebo data
# In Python_Norm_1m_Placebo every county is normalized to 100 on August 13, 2020. A Back to normal is required here. 
Wnorm=(wb0['Winnebago'].iloc[[101]]/100).values[0]   #Normalization coefficient for Winnebago
Brnorm=(wb0['Border'].iloc[[101]]/100).values[0]     #Normalization coef for bordering counties
Bnorm=(wb0['Boone'].iloc[[101]]/100).values[0]       #Normalization coefficient 
Dnorm=(wb0['DeKalb'].iloc[[101]]/100).values[0]      #Normalization coefficient 
Onorm=(wb0['Ogle'].iloc[[101]]/100).values[0]        #Normalization coefficient 
Snorm=(wb0['Steph'].iloc[[101]]/100).values[0]       #Normalization coefficient 

y_all = wb['Winnebago']                    #Define the Treatment/dependent variable
X_all = wb.drop(['Winnebago','Date','After_Treatment', 'Border'], axis = 1)    #Define the donor pool/indep var.
before_treat=wb['After_Treatment']==False
b= y_all[before_treat]
A= X_all[before_treat]

#This part calculates the synthetic control weights
from toolz import reduce, partial
def loss_w(W, X, y) -> float:
    return np.sqrt(np.mean((y - X.dot(W))**2))
lambda x: np.sum(x) - 1
from scipy.optimize import fmin_slsqp

def get_w(X, y):
    w_start = [1/X.shape[1]]*X.shape[1]
    weights = fmin_slsqp(partial(loss_w, X=X, y=y),
                         np.array(w_start),
                         f_eqcons=lambda x: np.sum(x) - 1,
                         bounds=[(0.0, 1.0)]*len(w_start),
                         disp=False)
    return weights
Winneb_weights = get_w(A, b)
print()
print("Winnebago synthetic weights are", Winneb_weights)
print("Winnebago Sum:", Winneb_weights.sum())
np.round(Winneb_weights, 4)
Winnebago_synth = X_all.values.dot(Winneb_weights)
Winnebago_norm=wb['Winnebago']*Wnorm
Diff_Synth=Winnebago_synth*Wnorm-Winnebago_norm

#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(A,b)
WOLS=model.predict(X_all)
Diff_OLS=WOLS*Wnorm-Winnebago_norm
print("The OLS coefficients of Winnebago are")
print('intercept', model.intercept_, 'slopes', model.coef_)
time=pd.to_datetime(wb['Date']).dt.date # takes out the time stamp from date

#Here we calculate the average deviation from counterfactuals after August 14.  
rows= len(Diff_Synth);  
sumRowS = 0
sumRowSb = 0
sumRowO = 0 
avSa=0
avSb=0
import math
for i in range(32,rows):  #32= august 14
    sumRowS = sumRowS + Diff_Synth[i]**2;   #Synthetic errors post treatment
    sumRowO = sumRowO + Diff_OLS[i]**2;     #OLS errors
avSa = sumRowS/(rows-32)
MSPE=math.sqrt(avSa)
avO = sumRowO/(rows-32)
MSPE_OLS=math.sqrt(avO)
for i in range(0,31):
    sumRowSb = sumRowSb + Diff_Synth[i]**2; #Synthetic errors pre treatment
avSb = sumRowSb/(32)
MSPEb=math.sqrt(avSb)
RMSPE=MSPE/MSPEb

print()
print('The Treatment date is',wb['Date'][31])
print('The RMSPE of the Synthetic Difference After Treatment is',"{:.2%}".format(MSPE))
print('The RMSE of the OLS Difference After Treatment is',"{:.2%}".format(MSPE_OLS))

#BOONE
y_all1 = wb['Boone']                    #Define the dependent variable
X_all1 = wb.drop(['Boone','Date','After_Treatment', 'Border'], axis = 1)    #Define the indep var.
before_treat=wb['After_Treatment']==False
b1= y_all1[before_treat]
A1= X_all1[before_treat]
Boone_weights = get_w(A1, b1)
print("Boone synthetic weights are", Boone_weights)
print("Boone Sum:", Boone_weights.sum())
np.round(Boone_weights, 4)
Boone_synth = X_all1.values.dot(Boone_weights)
Diff_Synth1=(Boone_synth-wb['Boone'])*Bnorm
#RMSPE Calculation
sumRowS1a = 0
sumRowS1b = 0
for i in range(32,rows):  #32= august 13
    sumRowS1a = sumRowS1a + Diff_Synth1[i]**2;
avS1a = sumRowS1a/(rows-32)
MSPE_1a=math.sqrt(avS1a)
for i in range(0,31):
    sumRowS1b = sumRowS1b + Diff_Synth1[i]**2;
avS1b = sumRowS1b/(32)
MSPE_1b=math.sqrt(avS1b)
RMSPE_1=MSPE_1a/MSPE_1b


#DEKALB
y_all2 = wb['DeKalb']                    #Define the dependent variable
X_all2 = wb.drop(['DeKalb','Date','After_Treatment', 'Border'], axis = 1)    #Define the indep var.
before_treat=wb['After_Treatment']==False
b2= y_all2[before_treat]
A2= X_all2[before_treat]
DeKalb_weights = get_w(A2, b2)
print("DeKalb synthetic weights are", DeKalb_weights)
print("DeKalb Sum:",DeKalb_weights.sum())
np.round(DeKalb_weights, 4)
DeKalb_synth = X_all2.values.dot(DeKalb_weights)
Diff_Synth2=(DeKalb_synth-wb['DeKalb'])*Dnorm
#RMSPE Calculation
sumRowS2a = 0
sumRowS2b = 0
for i in range(32,rows):  #32= august 13
    sumRowS2a = sumRowS2a + Diff_Synth2[i]**2;
avS2a = sumRowS2a/(rows-32)
MSPE_2a=math.sqrt(avS2a)
for i in range(0,31):
    sumRowS2b = sumRowS2b + Diff_Synth2[i]**2;
avS2b = sumRowS2b/(32)
MSPE_2b=math.sqrt(avS2b)
RMSPE_2=MSPE_2a/MSPE_2b

#OGLE
y_all3 = wb['Ogle']                    #Define the dependent variable
X_all3 = wb.drop(['Ogle','Date','After_Treatment', 'Border'], axis = 1)    #Define the indep var.
before_treat=wb['After_Treatment']==False
b3= y_all3[before_treat]
A3= X_all3[before_treat]
Ogle_weights = get_w(A3, b3)
print("Ogle synthetic weights are", Ogle_weights)
print("Ogle Sum:",Ogle_weights.sum())
np.round(Ogle_weights, 4)
Ogle_synth = X_all3.values.dot(Ogle_weights)
Diff_Synth3=(Ogle_synth-wb['Ogle'])*Onorm
#RMSPE Calculation
sumRowS3a = 0
sumRowS3b = 0
for i in range(32,rows):  #32= august 13
    sumRowS3a = sumRowS3a + Diff_Synth3[i]**2;
avS3a = sumRowS3a/(rows-32)
MSPE_3a=math.sqrt(avS3a)
for i in range(0,31):
    sumRowS3b = sumRowS3b + Diff_Synth3[i]**2;
avS3b = sumRowS3b/(32)
MSPE_3b=math.sqrt(avS3b)
RMSPE_3=MSPE_3a/MSPE_3b

#STEPHENSON
y_all4 = wb['Steph']                    #Define the dependent variable
X_all4 = wb.drop(['Steph','Date','After_Treatment', 'Border'], axis = 1)    #Define the indep var.
before_treat=wb['After_Treatment']==False
b4= y_all4[before_treat]
A4= X_all4[before_treat]
Steph_weights = get_w(A4, b4)
print("Steph synthetic weights are", Steph_weights)
print("Steph Sum:", Steph_weights.sum())
np.round(Steph_weights, 4)
Steph_synth = X_all4.values.dot(Steph_weights)
Diff_Synth4=(Steph_synth-wb['Steph'])*Snorm
#RMSPE Calculation
sumRowS4a = 0
sumRowS4b = 0
for i in range(32,rows):  #32= august 13
    sumRowS4a = sumRowS4a + Diff_Synth4[i]**2;
avS4a = sumRowS4a/(rows-32)
MSPE_4a=math.sqrt(avS4a)
for i in range(0,31):
    sumRowS4b = sumRowS4b + Diff_Synth4[i]**2;
avS4b = sumRowS4b/(32)
MSPE_4b=math.sqrt(avS4b)
RMSPE_4=MSPE_4a/MSPE_4b

RMSPE_matrix=np.zeros(5)
RMSPE_matrix[0]=RMSPE
RMSPE_matrix[1]=RMSPE_1
RMSPE_matrix[2]=RMSPE_2
RMSPE_matrix[3]=RMSPE_3
RMSPE_matrix[4]=RMSPE_4

# PLACEBO LINEAR REGRESSIONS
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(A1,b1)
BOLS=model.predict(X_all1)
Boone_norm=wb['Boone']*Bnorm
Diff_OLS1=BOLS*Bnorm-Boone_norm

model.fit(A2,b2)
DOLS=model.predict(X_all2)
DeKalb_norm=wb['DeKalb']*Dnorm
Diff_OLS2=DOLS*Dnorm-DeKalb_norm

model.fit(A3,b3)
OOLS=model.predict(X_all3)
Ogle_norm=wb['Ogle']*Onorm
Diff_OLS3=OOLS*Onorm-Ogle_norm

model.fit(A4,b4)
SOLS=model.predict(X_all4)
Steph_norm=wb['Steph']*Snorm
Diff_OLS4=SOLS*Snorm-Steph_norm

#First Figure compares the vaccination rates in Winnebago vs 
#all bordering counties as one entity
plt.figure(figsize=(10,6))
plt.title ('Figure 1: Vaccinations Rates of 12-18 Age Population')
plt.ylabel("Vaccination Rates")
plt.plot(time, Winnebago_norm, label="Winnebago",lw=2, c='0.15') #color='red'
plt.plot(time, wb['Border']*Brnorm,linestyle="-.", c='0.45' , label="Neighboring Counties") #color='blue'
plt.vlines(x='2021-08-13', ymin=0.32, ymax=0.50,color='black', linestyle="--", lw=2)
plt.text(x='2021-08-06',y=0.31,s="Gift Card Announcement")
plt.grid(True,linestyle="--")
plt.legend(loc='upper left');
plt.xticks(['2021-07-13','2021-07-29', '2021-08-13','2021-08-29', '2021-09-13'],rotation = 40)


#Second Figure: Real Winnebago, OLS Winnebago, Synthetic Winnebago
plt.figure(figsize=(10,6))
plt.title ('Figure 2: Winnebago County, synthetic Winnebago County and Winnebago County with linear regression. ')
plt.plot(time,Winnebago_norm, label="Winnebago",c='0.10',) #color='red')
plt.plot(time, WOLS*Wnorm,linestyle="--",c='0.35', label="Regression") #color='orange'
plt.plot(time, Winnebago_synth*Wnorm, linestyle="-.",c='0.60' ,label="Synthetic Control") #color='green'
plt.ylabel("Vaccination Rates")
plt.vlines(x='2021-08-13', ymin=0.31, ymax=0.45, color='black', linestyle="--", lw=2)
plt.text(x='2021-08-06',y=0.305,s="Gift Card Announcement")
plt.xticks(rotation = 40)
plt.grid(True,linestyle="--")
plt.legend();
plt.xticks(['2021-07-13','2021-07-29', '2021-08-13','2021-08-29', '2021-09-13'])

#Third Figure calculates and draws the difference between estimations and real
plt.figure(figsize=(10,6))
plt.ylabel("Change in Vaccination Rates")
plt.plot(time,Diff_OLS, c='0.35', linestyle="--", label="Regression Gap") #color='orange',
plt.plot(time,Diff_Synth,linestyle="-.", c='0.60', label="Synthetic Control Gap") #color='green'
plt.vlines(x='2021-08-13', ymin=-0.025, ymax=0.005, color='black',linestyle="--",lw=2)
plt.text(x='2021-08-05',y=-0.0255,s="Gift Card Announcement")
plt.xticks(['2021-07-13','2021-07-29', '2021-08-13','2021-08-29', '2021-09-13'])
plt.xticks(rotation = 40)
plt.grid(True,linestyle="--")
plt.legend(loc='lower left');
plt.title ('Vaccinations Differences')

#Fifth Graph Synthetic Placebo Differences
fig, (ax1, ax2) = plt.subplots(2,figsize=(9,6))
#fig.suptitle('PLACEBO ANALYSIS')

ax1.set_title('Fig 5: Vaccination rates gaps in Winnebago County and placebo gaps in all 4 control counties')
ax1.plot(time,Diff_Synth,c='0.05',linewidth=2, label='Winnebago') #color='green'
ax1.plot(time,Diff_Synth1,color='black',linestyle='dashdot',linewidth=1,label='Control counties')
ax1.plot(time,Diff_Synth2,color='black',linestyle='dashdot',linewidth=1)
ax1.plot(time,Diff_Synth3,color='black',linestyle='dashdot',linewidth=1)
ax1.plot(time,Diff_Synth4,color='black',linestyle='dashdot',linewidth=1)
ax1.set_ylabel("Gap in vaccination rates")
ax1.vlines(x='2021-08-13',ymin=-0.029, ymax=0.03, color='black', linestyle='dashed',lw=2)
ax1.tick_params('x',labelrotation=30)
ax1.set_xticks(['2021-07-13','2021-07-29', '2021-08-13','2021-08-29', '2021-09-13'])
ax1.grid(linestyle="--")
ax1.legend(loc='lower left')
ax1.text(x='2021-08-07',y=-0.030,s="Gift Card Announcement")
fig.tight_layout(h_pad=3.0)
#AX2:
ax2.set_title('')  #Ratio of Post-Announcement RMSPE to Pre-Announcement RMSPE'
County = ['Winnebago','Boone','DeKalb','Ogle','Stephenson']
ax2.bar(County,RMSPE_matrix,color='black', label="RMSE") #color='orange'
plt.xticks(rotation=35)
ax2.set_ylabel("RMSPE")
plt.grid(True,linestyle="--",linewidth=0.6)

print()
print('Winnebago MSPE is',"{:.6}".format(RMSPE))
print('Boone MSPE is',"{:.2}".format(RMSPE_1))
print('DeKalb MSPE is',"{:.3}".format(RMSPE_2))
print('Ogle MSPE is',"{:.4}".format(RMSPE_3))
print('Stephenson MSPE is',"{:.2}".format(RMSPE_4))
"""
#Print descroptive statistics
DS=np.empty((8,5))
for i in range(5):
   wb.iloc[:,i+1].describe()
print(DS)
"""