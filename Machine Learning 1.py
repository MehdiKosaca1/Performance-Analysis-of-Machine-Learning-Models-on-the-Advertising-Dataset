#!/usr/bin/env python
# coding: utf-8

# # Basit Doğrusal Regresyon

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194523.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194523.png)![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194433.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194433.png)

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("C:\\Users\\mehdi\\Downloads\\Advertising.csv")


# In[3]:


data.head()


# In[4]:


data = data.iloc[:,1:len(data)]


# In[5]:


import seaborn as sns
sns.jointplot(x = "TV", y= "sales", data= data, kind="reg");


# In[6]:


from sklearn.linear_model import LinearRegression


# In[7]:


x = data[["TV"]]


# In[8]:


y = data[["sales"]]


# In[9]:


reg = LinearRegression()


# In[10]:


model = reg.fit(x,y)


# In[11]:


model2 = LinearRegression().fit(data[["TV"]],data[["sales"]])


# In[12]:


model2.coef_
model.intercept_


# In[13]:


model


# In[14]:


str(model)


# In[15]:


# b0
model.intercept_


# In[16]:


# b1
model.coef_


# In[17]:


# R kare
model.score(x,y)


# In[18]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.regplot(x=data["TV"], y=data["sales"], ci=None, scatter_kws={'color': 'r', 's': 9})
g.set_title("Model Denklemi: Sales = 7.03+ TV*0.05")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)


# In[19]:


model.predict([[165]])


# In[20]:


model.predict([[365]])


# In[21]:


def ml(x):
    print("modelin b0:")
    print(x.intercept_)
    print("modelin b1:")
    print(x.coef_)


# In[22]:


ml(model)


# In[23]:


gercek_y = y[0:10]
tahmin_edilen_y = pd.DataFrame(model.predict(x)[0:10])


# In[24]:


hatalar = pd.concat([gercek_y,tahmin_edilen_y],axis=1)
hatalar.columns= ["gercek_y","tahmin_edilen_y"]


# In[25]:


hatalar


# In[26]:


hatalar["hata"] = hatalar["gercek_y"] - hatalar["tahmin_edilen_y"]


# In[27]:


hatalar


# In[28]:


hatalar["hata_kareler"] = hatalar["hata"]**2


# In[29]:


hatalar


# In[30]:


import numpy as np
np.mean(hatalar["hata_kareler"])


# In[31]:


np.sqrt(np.mean(hatalar["hata_kareler"]))


# # Çoklu Doğrusal Regresyon

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194623.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194623.png)![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194725.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194725.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194825.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194825.png)![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194857.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194857.png)

# In[32]:


import pandas as pd
data2 = pd.read_csv("C:\\Users\\mehdi\\Downloads\\Advertising.csv")


# In[33]:


data2 = data2.iloc[:,1:len(data2)]
data2.head()


# In[34]:


x= data2.drop("sales",axis=1)
y= data2[["sales"]]


# In[35]:


x.head()


# In[38]:


y.head()


# In[39]:


#Statsmodels ile model kurma


# In[40]:


import statsmodels.api as sm


# In[41]:


modeld = sm.OLS(y,x).fit()


# In[42]:


modeld.summary()


# In[43]:


lm = sm.OLS(y,x)


# In[44]:


model2 = lm.fit()


# In[45]:


model2.summary()


# In[46]:


#scikit learn ile model kurmak


# In[47]:


from sklearn.linear_model import LinearRegression


# In[48]:


model = LinearRegression().fit(x,y)


# In[49]:


model.intercept_


# In[50]:


model.coef_


# In[51]:


yeni_veri= [[30],[10],[40]]


# In[52]:


import pandas as pd
yeni_veri = pd.DataFrame(yeni_veri).T


# In[53]:


yeni_veri


# In[54]:


model.predict(yeni_veri)


# In[55]:


from sklearn.metrics import mean_squared_error


# In[56]:


import numpy as np


# In[57]:


MSE = mean_squared_error(y,model.predict(x)) #hata kareler ortalaması


# In[58]:


RMSE =np.sqrt(MSE) #hata kareler ortalaması karakökü


# In[59]:


MSE


# In[60]:


RMSE


# In[61]:


model.coef_


# In[62]:


model.intercept_


# In[63]:


#sınama seti
from sklearn.model_selection import train_test_split


# In[64]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.20, random_state= 99)


# In[65]:


x_train.head()


# In[66]:


y_train.head()


# In[67]:


x_test.head()


# In[68]:


y_test.head()


# In[69]:


model = LinearRegression().fit(x_train,y_train)


# In[70]:


#eğitim hatası
np.sqrt(mean_squared_error(y_train, model.predict(x_train)))


# In[71]:


#test hatası
np.sqrt(mean_squared_error(y_test, model.predict(x_test)))


# In[72]:


#k-katlı cv
from sklearn.model_selection import cross_val_score


# In[73]:


#cv rmse
np.sqrt(np.mean(-cross_val_score(model,x_train,y_train, cv= 10 ,scoring= "neg_mean_squared_error")))


# In[74]:


#cv rmse
np.sqrt(np.mean(-cross_val_score(model,x,y, cv= 10 ,scoring= "neg_mean_squared_error")))


# # Ridge Regresyon

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20195004.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20195004.png)![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194939.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20194939.png)

# In[75]:


# bu alanda kullanılan kütüphaneler
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV


# In[76]:


data = pd.read_csv("C:\\Users\\mehdi\\Downloads\\Hitters.csv")


# In[77]:


data.head()


# In[78]:


data=data.dropna()
dms = pd.get_dummies(data[['League', 'Division', 'NewLeague']])

y = data["Salary"]

x_ = data.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

x = pd.concat([x_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[79]:


data.head()


# In[80]:


data.shape


# In[81]:


ridge_model = Ridge(alpha=0.1).fit(x_train,y_train)


# In[82]:


ridge_model


# In[83]:


ridge_model.coef_


# In[84]:


redige_model= Ridge().fit(x_train,y_train)


# In[85]:


y_pred = ridge_model.predict(x_train)


# In[86]:


y_pred[0:10]


# In[87]:


y_train[0:10]


# In[88]:


RMSE = np.sqrt(mean_squared_error(y_train,y_pred))
RMSE


# In[89]:


np.sqrt(np.mean(-cross_val_score(ridge_model,x_train,y_train, cv= 10 , scoring="neg_mean_squared_error")))


# In[90]:


#test hatası
y_pred = ridge_model.predict(x_train)


# In[94]:


# RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
# RMSE


# In[95]:


ridge_model=Ridge(1000).fit(x_train,y_train)


# In[96]:


y_pred = ridge_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[97]:


lambdalar1= np.random.randint(0,1000,100)


# In[98]:


lambdalar2= 10**np.linspace(10,-2,100)*0.5


# In[99]:


from sklearn.linear_model import RidgeCV


# In[100]:


# ridgecv =RidgeCV(alphas=lambdalar2,scoring="neg_squared_error",cv=10,normalize= True) RidgeCv fonksiyonun da normalize kalkmış


# In[101]:


from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# lambdalar2'nin tanımlı olduğundan ve doğru bir değer dizisi içerdiğinden emin olun
ridgecv = RidgeCV(alphas=lambdalar1, scoring="neg_mean_squared_error", cv=10)

# Veri normalleştirme ile bir pipeline oluşturun
pipeline = make_pipeline(StandardScaler(), ridgecv)

# Pipeline'ı kullanarak modelinizi eğitin
# X_train ve y_train, eğitim veri setinizin özellikleri ve hedef değişkeni olmalıdır
# pipeline.fit(X_train, y_train)


# In[102]:


ridgecv.fit(x_train,y_train)


# In[103]:


ridgecv.alpha_


# In[104]:


ridge_tuned=Ridge(alpha= ridgecv.alpha_).fit(x_train,y_train)


# In[105]:


y_pred = ridge_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# # Lasso Regresyon

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20195050.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-17%20195050.png)

# In[106]:


# bu alanda kullanılan kütüphaneler
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge , Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV


# In[107]:


data = pd.read_csv("C:\\Users\\mehdi\\Downloads\\Hitters.csv")
data=data.dropna()
dms = pd.get_dummies(data[['League', 'Division', 'NewLeague']])

y = data["Salary"]

x_ = data.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

x = pd.concat([x_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[108]:


data.head()


# In[109]:


data.shape


# In[110]:


lasso_model=Lasso().fit(x_train,y_train)


# In[111]:


lasso_model.intercept_


# In[112]:


lasso_model.coef_


# In[115]:


#farklı lambda degerlerine karsilik katsayilar
lasso= Lasso()
coefs= []
alphas = np.random.randint(0,100000,100)
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(x_train,y_train)
    coefs.append(lasso.coef_)
    


# In[116]:


ax = plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale("log")


# In[117]:


lasso_model.alpha


# In[118]:


lasso_model.predict(x_train)[0:5]


# In[119]:


lasso_model.predict(x_test)[0:5]


# In[120]:


y_pred = lasso_model.predict(x_test)


# In[121]:


from sklearn.metrics import mean_squared_error


# In[122]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[123]:


r2_score(y_test,y_pred)


# In[124]:


lasso_cv_model = LassoCV(cv=10,max_iter=100000).fit(x_train,y_train)


# In[125]:


lasso_cv_model.alpha_


# In[126]:


lasso_tuned= Lasso(alpha=lasso_cv_model.alpha_).fit(x_train,y_train)


# In[127]:


y_pred = lasso_tuned.predict(x_test)


# In[128]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[129]:


pd.Series(lasso_tuned.coef_,index=x_train.columns)


# # ElasticNet Regresyon

# In[130]:


# bu alanda kullanılan kütüphaneler
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge , Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV


# In[131]:


data = pd.read_csv("C:\\Users\\mehdi\\Downloads\\Hitters.csv")
data=data.dropna()
dms = pd.get_dummies(data[['League', 'Division', 'NewLeague']])

y = data["Salary"]

x_ = data.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

x = pd.concat([x_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[132]:


enet_model= ElasticNet().fit(x_train,y_train)


# In[133]:


enet_model.coef_


# In[134]:


enet_model.intercept_


# In[135]:


#tahmin
enet_model.predict(x_train)[0:10]


# In[136]:


enet_model.predict(x_test)[0:10]


# In[137]:


y_pred= enet_model.predict(x_test)


# In[138]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[139]:


#R kare hesaplanması
r2_score(y_test,y_pred)


# In[140]:


enet_cv_model = ElasticNetCV(cv=10).fit(x_train,y_train)


# In[141]:


enet_cv_model.alpha_


# In[142]:


enet_cv_model.intercept_


# In[143]:


enet_cv_model.coef_


# In[144]:


#final modeli
enet_tuned = ElasticNet(alpha=enet_cv_model.alpha_).fit(x_train,y_train)


# In[145]:


y_pred= enet_tuned.predict(x_test)


# In[146]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[147]:


get_ipython().run_line_magic('pinfo', 'ElasticNet')


# In[ ]:




