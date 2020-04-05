#!/usr/bin/env python
# coding: utf-8

# # EDA comparing TRAIN.CSV and Predict.CSV

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

import plotly.offline as py 
from plotly.offline import init_notebook_mode, iplot
py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot
import seaborn as sns


# # TRIAN.CSV　２月１日〜２月２８日

# In[2]:


df_train = pd.read_csv("Data/train.csv", parse_dates=["datetime"]).set_index("datetime")
print(df_train.columns.tolist())


# In[3]:


df_train.head()


# In[4]:


# 約40万ポイント
df_train.shape


# In[5]:


df_train.index


# In[6]:


dateFeb = pd.date_range(start = '2019-02-01 00:00', end='2019-02-28 23:59',  freq='S')
dateFeb


# In[7]:


per_feb = (len(df_train.index)/len(dateFeb)) *100

print("# of data points in df_train")
print(len(df_train.index))
print("# of data points in Feb in sec")
print(len(dateFeb))
print()

print("２０１９年２月分のデータから {}".format(round(per_feb)), '%がサンプルされている')


# # PREDICT.CSV　４月１日〜４月３０日

# In[8]:


df_predict = pd.read_csv("Data/predict.csv", parse_dates=["datetime"]).set_index("datetime")
print(df_predict.columns.tolist())


# In[9]:


df_predict.head()


# In[10]:


# 約２７万
df_predict.shape


# In[11]:


df_predict.index


# In[12]:


dateApril = pd.date_range(start = '2019-04-01 00:00', end='2019-04-30 23:59',  freq='S')
dateApril


# In[13]:


per_april = (len(df_predict.index)/len(dateApril)) *100

print("# of data points in df_train")
print(len(df_predict.index))
print("# of data points in April in sec")
print(len(dateApril))
print()

print("２０１９年４月分のデータから {}".format(round(per_april)), '%がサンプルされている')


# # Comp summary between Train.csv and Predict.csv

# In[14]:


df_train.describe()


# In[15]:


df_predict.describe()


# mean values seems different between train.csv and predict.csv
# 
# ai0 and ai1 max values are different between train.csv and predict.csv
# 
# ai3 might have outliers

# # Viszualization

# In[17]:


# Train.csv
def timeplot(df):
    for col in df.columns.tolist():
        fig = plt.figure(figsize=(16,4))
        plt.plot(df.index,df[col].values)
        plt.title(col)
        plt.show()
        
timeplot(df_train)


# In[18]:


# predict
timeplot(df_predict)


# # Correlation

# In[42]:


def corr_plot(df, title='Dataset'):

    fig, ax = plt.subplots(1,1, figsize = (15,6))
    hm = sns.heatmap(df.iloc[:,:].corr(),
                    ax = ax,
                    cmap = 'coolwarm',
                    annot = True,
                    fmt = '.2f',
                    linewidths = 0.05)

    fig.subplots_adjust(top=0.93)
    fig.suptitle(title, 
                  fontsize=14, 
                  fontweight='bold')
    
corr_plot(df_train, title= 'train dataset')


# In[43]:


corr_plot(df_predict, title= 'Predict dataset')


# In[44]:


# Only train.csv
plt.figure(figsize=(26, 16))
for i, col in enumerate(df_train.columns):
    ax = plt.subplot(4, 3, i + 1)
    sns.distplot(df_train[col], bins=100, label='train')
    ax.legend()  


# In[45]:


# both train.csv and predict.csv
plt.figure(figsize=(26, 16))
for i, col in enumerate(df_predict.columns):
    ax = plt.subplot(3, 3, i + 1)
    sns.distplot(df_train[col].values, bins=100, label='train')
    sns.distplot(df_predict[col].values, bins=100, label='predict')
    ax.legend()  


# # Checking Scored (Target) values  

# In[46]:


target = df_train['Scored'].value_counts().reset_index().rename(columns = {'index' : 'target'})
target


# In[47]:


trace0 = go.Bar(
    x = df_train['Scored'].value_counts().index,
    y = df_train['Scored'].value_counts().values
    )

trace1 = go.Pie(
    labels = df_train['Scored'].value_counts().index,
    values = df_train['Scored'].value_counts().values,
    domain = {'x':[0.55,1]})

data = [trace0, trace1]
layout = go.Layout(
    title = 'Frequency Distribution for surface/target data',
    xaxis = dict(domain = [0,.50]))

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# # Data Imputation 

# In[78]:


# Missingness plot
for i in range(0, len(df_train)-100, 1000):
    fig, ax = plt.subplots(1,1, figsize = (20,16))
    sns.heatmap(df_train.iloc[i: i + 1000,].isnull(), cbar=False)
    plt.savefig('missing_plot/' + str(i) + '.png')
    #plt.show()
    plt.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df_train["Frequency"] = df["Frequency"].fillna(method = "bfill")


# In[ ]:





# # 新しいコラムを追加する際の論点
# 
# ##### 〇. 頻度の設定値と実際値の差は、どの程度か？
# 1. その差は、異常判定にどの程度影響しているのか？（フラグを立てる必要はあるのか？） 
# 2. 差があった場合、65%は、異常の運転と出ている
# 3. なので、フラグを付けるべき

# ### 頻度について

# In[ ]:


df["diff_freq"] =  df["FrequencyOutput"] - df["Frequency"]
df["diff_temp"] =  df["TempSV"] - df["TempPV"]


df_diff_exist = df[df["diff_freq"] != 0]
df_diff_exist_normal  = df_diff_exist[df_diff_exist["Scored"] == 1]
df_diff_exist_anomaly = df_diff_exist[df_diff_exist["Scored"] == -1]

print("頻度：設定値と実際値が離れている割合: {}".format(len(df[df["diff_freq"] != 0]) / len(df) * 100))
print("そのうち、異常が出ている割合:{}".format(len(df_diff_exist_anomaly)/len(df_diff_exist)*100))


# ### 温度について

# In[ ]:


df["diff_temp"] =  df["TempPV"] - df["TempSV"]
df_diff_exist_temp = df[df["diff_temp"] != 0]
df_diff_exist_normal_temp  = df_diff_exist[df_diff_exist["Scored"] == 1]
df_diff_exist_anomaly_temp = df_diff_exist[df_diff_exist["Scored"] == -1]

print("温度：設定値と実際値が離れている割合: {}".format(len(df[df["diff_temp"] != 0]) / len(df) * 100))
print("そのうち、異常が出ている割合:{}".format(len(df_diff_exist_anomaly_temp)/len(df_diff_exist_normal_temp)*100))


# # 欠損値処理の処理の論点
# ##### 〇. 頻度に関して、一、二、三秒単位の差分はいくつか？

# In[ ]:


df_frequency = df[["Frequency", "FrequencyOutput"]]
df_frequency["diff_one_second"] = df_frequency["Frequency"] - df_frequency["Frequency"].shift(1)
df_frequency["diff_one_second"].plot(kind="hist")
plt.show()

df_frequency["diff_two_second"] = df_frequency["FrequencyOutput"] - df_frequency["FrequencyOutput"].shift(1)
df_frequency["diff_two_second"].plot(kind="hist")
plt.show()


# In[ ]:


df["Frequency"] = df["Frequency"].fillna(method = "bfill")
df["FrequencyOutput"] = df["FrequencyOutput"].fillna(method="bfill")


# In[ ]:


df["diff_freq"] =  df["FrequencyOutput"] - df["Frequency"]


# In[ ]:


df[["ai0", "ai1", "ai2", "ai3"]] = df[["ai0", "ai1", "ai2", "ai3"]].fillna(method="bfill")


# In[ ]:


df = df.fillna(method="bfill")


# In[ ]:


# フラグの挿入
def put_flag(num):
    if num != 0:
        return 1
    else:
        return 0


df["flag_diff_exist"] = df["diff_freq"].apply(put_flag)


# In[ ]:


df.to_csv(r"C:\Users\sasdemo01\Desktop\Progress\dataset\train_bfill_flag_added.csv")


# # 新しいコラムの追加

# In[ ]:


new_col_1 = []
for col in df.columns.tolist():
    new_col_1.append(col + "_t-1")

new_col_2 = []
for col in df.columns.tolist():
    new_col_2.append(col + "_t-2")

new_col_3 = []
for col in df.columns.tolist():
    new_col_3.append(col + "_t-3")

new_col_4 = []
for col in df.columns.tolist():
    new_col_4.append(col + "_t-4")
    
new_col_5 = []
for col in df.columns.tolist():
    new_col_5.append(col + "_t-5")


# In[ ]:




