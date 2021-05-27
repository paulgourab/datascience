#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)


# In[3]:


df1=pd.read_csv("FinalProject2.csv")
df1.head()


# In[3]:


df1.shape


# In[4]:


df1.groupby('area_type')['area_type'].agg('count')


# In[5]:


df2=df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.head()


# In[6]:


df2.isnull().sum()


# In[7]:


df3=df2.dropna()
df3.isnull().sum()


# In[8]:


df3.shape


# In[9]:


df3['size'].unique()


# In[10]:


df3['bhk']=df3['size'].apply(lambda x: x.split(' ')[0])
df3.head()


# In[11]:


df3['bhk'].unique()


# In[12]:


df3.total_sqft.unique()


# In[13]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[15]:


df3[~df3['total_sqft'].apply(is_float)].head()


# In[16]:


def convert_sqft_to_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return(float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[17]:


df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()


# In[18]:


df4.loc[250]


# In[19]:


df4.head(10)


# In[20]:


df5=df4.copy()
df5['rent_per_sqft']=df5['price']/df5['total_sqft']
df5.head()


# In[21]:


df5.location.unique()


# In[22]:


len(df5.location.unique())


# In[23]:


df5.location=df5.location.apply(lambda x: x.strip())
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[30]:


len(location_stats[location_stats<=5])


# In[25]:


location_less_than_5 = location_stats[location_stats<=5]
location_less_than_5


# In[26]:


len(df5.location.unique())


# In[27]:


df5.location=df5.location.apply(lambda x: 'other' if x in location_less_than_5 else x)
len(df5.location.unique())


# In[28]:


df5.head(10)


# In[36]:


df5.shape


# In[40]:


type(df5.bhk)


# In[48]:


df5.rent_per_sqft.describe()


# In[50]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.rent_per_sqft)
        st = np.std(subdf.rent_per_sqft)
        reduced_df = subdf[(subdf.rent_per_sqft>(m-st)) & (subdf.rent_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df6 = remove_pps_outliers(df5)
df6.shape


# In[67]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=10)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=10)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Rent (Thousand Bangladesh Taka)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df6,'sec 1')


# In[68]:


plot_scatter_chart(df6,'sec 2')


# In[69]:


plot_scatter_chart(df6,'sec 3')


# In[70]:


plot_scatter_chart(df6,'sec 4')


# In[71]:


plot_scatter_chart(df6,'sec 6')


# In[72]:


plot_scatter_chart(df6,'sec 7')


# In[73]:


plot_scatter_chart(df6,'sec 8')


# In[74]:


plot_scatter_chart(df6,'sec 10')


# In[75]:


plot_scatter_chart(df6,'sec 11')


# In[76]:


plot_scatter_chart(df6,'sec 12')


# In[77]:


plot_scatter_chart(df6,'sec 13')


# In[78]:


plot_scatter_chart(df6,'sec 14')


# In[79]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df6.rent_per_sqft,rwidth=0.8)
plt.xlabel("Rent Per Square Feet")
plt.ylabel("Count")


# In[80]:


df6.bath.unique()


# In[81]:


plt.hist(df6.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[85]:


df6[df6.bath>4]


# In[91]:


dummies = pd.get_dummies(df6.location)
dummies.head(3)


# In[97]:


df12 = df6.drop('location',axis='columns')
df12.head(2)


# In[98]:


df12.shape


# In[99]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[100]:


X.shape


# In[109]:


y=df12.price
y.head(3)


# In[113]:


len(y)


# In[114]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[116]:


from sklearn.linear_model import LinearRegression
try:
    lr_clf = LinearRegression()
    lr_clf.fit(X_train,y_train)
    lr_clf.score(X_test,y_test)
except:
    print("Nothing")


# In[118]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

try:
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    cross_val_score(LinearRegression(), X, y, cv=cv)
except:
    print("Nothing")


# In[119]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[120]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[122]:


predict_price('sec 2',1000, 2, 3)


# In[ ]:




