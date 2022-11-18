#!/usr/bin/env python
# coding: utf-8

# # import required libray

# In[33]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# In[34]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='XeaAWDu0NTFq6Wd1KkU_9kiMsC5f8bAki0X4MI77fHRP',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'ibmdepolyment-donotdelete-pr-mut3vz0xh6wsjn'
object_key = 'PM_train.xlsx'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']

dataset_train = pd.read_excel(body.read())
dataset_train
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='XeaAWDu0NTFq6Wd1KkU_9kiMsC5f8bAki0X4MI77fHRP',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'ibmdepolyment-donotdelete-pr-mut3vz0xh6wsjn'
object_key = 'PM_train.xlsx'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']

dataset_train = pd.read_excel(body.read())
dataset_train.head()


# In[36]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='XeaAWDu0NTFq6Wd1KkU_9kiMsC5f8bAki0X4MI77fHRP',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'ibmdepolyment-donotdelete-pr-mut3vz0xh6wsjn'
object_key = 'PM_test.xlsx'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']

dataset_test = pd.read_excel(body.read())
dataset_test.head()


# In[37]:



import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='XeaAWDu0NTFq6Wd1KkU_9kiMsC5f8bAki0X4MI77fHRP',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'ibmdepolyment-donotdelete-pr-mut3vz0xh6wsjn'
object_key = 'PM_truth.xlsx'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']

pm_truth = pd.read_excel(body.read())
pm_truth.head()


# In[38]:


rul=pd.DataFrame(dataset_test.groupby('id')['cycle'].max()).reset_index()
rul.columns=['id','max']
rul.head()


# In[39]:


pm_truth['rtf']=pm_truth['more']+rul['max']
pm_truth.head()


# In[40]:


pm_truth.drop('more',axis=1,inplace=True)
dataset_test=dataset_test.merge(pm_truth,on=['id'],how='left')
dataset_test['ttf']=dataset_test['rtf']-dataset_test['cycle']
dataset_test.drop('rtf',axis=1,inplace=True)
dataset_test.head()


# In[41]:


dataset_train['ttf']=dataset_train.groupby(['id'])['cycle'].transform(max)-dataset_train['cycle']
dataset_train.head()


# In[42]:


df_train=dataset_train.copy()
df_test=dataset_test.copy()
period=30
df_train['label_bc']=df_train['ttf'].apply(lambda x: 1 if x <= period else 0)
df_test['label_bc']=df_test['ttf'].apply(lambda x: 1 if x <= period else 0)
df_train.head()


# In[43]:


x_train=df_train.iloc[:,:-1].values
y_train=df_train.iloc[:,-1:].values


# In[44]:


#Splitting data into train  and test
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.2,random_state=0)


# In[45]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[46]:


y_pred1=lr.predict(x_test)
from sklearn.metrics import accuracy_score
log_reg=accuracy_score(y_test,y_pred1)
log_reg


# In[47]:


from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y_pred1)
cm1


# In[48]:


import joblib
joblib.dump(lr, "engine_model.sav")


# In[49]:


get_ipython().system('pip install ibm_watson_machine_learning')


# In[76]:


from ibm_watson_machine_learning import APIClient
wml_credentials={
    "url":"https://us-south.ml.cloud.ibm.com",
    "apikey" :"lXz0XO1eh1nLZZLQ2m7wahlFW812sFxvI5d80PVEQAz9"
}
client=APIClient(wml_credentials)


# In[79]:


def guid_from_space_name(client, space_name):
    space = client.spaces.get_details()
    return(next(item for item in space['resources'] if item ['entity']['name'] == space_name)['metadata']['id'])


# In[80]:


space_uid = guid_from_space_name (client, 'model')
print("Space UID ="+ space_uid)
client.set.default_space(space_uid)


# In[83]:


client.software_specifications.list()


# In[88]:


software_spec_uid = client.software_specifications.get_uid_by_name("runtime-22.1-py3.9")
software_spec_uid


# In[93]:


model_details = client.repository.store_model(model=lr,meta_props ={
    client.repository.ModelMetaNames.NAME:"engine_model",
    client.repository.ModelMetaNames.TYPE:"scikit-learn_1.0",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_spec_uid
    
})
model_id=client.repository.get_model_uid(model_details)


# In[94]:


model_id


# In[95]:


x_train[0]


# In[97]:


lr.predict([[ 2.70000e+01,  1.08000e+02, -4.80000e-03,  0.00000e+00,
        1.00000e+02,  5.18670e+02,  6.42500e+02,  1.58658e+03,
        1.41020e+03,  1.46200e+01,  2.16100e+01,  5.53290e+02,
        2.38810e+03,  9.07818e+03,  1.30000e+00,  4.74700e+01,
        5.21280e+02,  2.38810e+03,  8.15268e+03,  8.44090e+00,
        3.00000e-02,  3.94000e+02,  2.38800e+03,  1.00000e+02,
        3.87300e+01,  2.33405e+01,  4.80000e+01]])


# In[ ]:




