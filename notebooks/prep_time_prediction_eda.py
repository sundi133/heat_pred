#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.ensemble import RandomForestRegressor
from hyperopt import tpe,hp,Trials
from hyperopt.fmin import fmin
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import os
import datetime
import numpy as np
data = pd.read_json('orders.json')


# In[2]:



def extract_order_item_len(row):
    ordered_items = row.obfuscated_item_names[1:-1]
    items = [x.strip() for x in ordered_items.split(',')]
    return len(items)

def extract_order_item(row, max_items):
    ordered_items = row.obfuscated_item_names[1:-1]
    items = [x.strip() for x in ordered_items.split(',')]
    items.sort()
    order_len = len(items)
    items_in_order=[]
    items_in_order.append(order_len)
    for i in range(max_items):
        if i < len(items):
            items_in_order.append(items[i])
        else:
            items_in_order.append(None)
    return tuple(items_in_order)


def read_csv_data(file):
    print("Reading progress...")
    return pd.read_csv(file)

def read_json_data(file):
    print("Reading progress...")
    return pd.read_json(file)

def feature_engg(data, feature_columns):
    print("Feature Engg in progress...")
    #dml_log.distml_logit({"status":"feature_engg"})

    data = data[feature_columns]
    data.set_index('activated_at_local', inplace=True)
    data.sort_values(by=['activated_at_local'], inplace=True, ascending=True)
    data = data.reset_index()

    derived_column_orders = []
    max_items = 3
    derived_column_orders.append('order_len')
    #for i in range(max_items):
    #    derived_column_orders.append('obfuscated_item_names_'+str(i))

    #data[derived_column_orders] = data.apply(extract_order_item, args=(max_items,),axis=1,result_type="expand")
    data[derived_column_orders] = data.apply(extract_order_item_len, axis=1, result_type="expand")
    #time dimension features
    data['scaled_subtotal'] = np.log(data['subtotal']+1)
    data['scaled_order_len'] = np.log(data['order_len']+1)
    data['Date'] = pd.to_datetime(data['activated_at_local'])
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d-%H:%M:%S')
    data['year'] = pd.DatetimeIndex(data['Date']).year
    data['month'] = pd.DatetimeIndex(data['Date']).month
    data['day'] = pd.DatetimeIndex(data['Date']).day
    data['hour'] = pd.DatetimeIndex(data['Date']).hour
    data['dayofyear'] = pd.DatetimeIndex(data['Date']).dayofyear
    data['weekofyear'] = pd.DatetimeIndex(data['Date']).weekofyear
    data['weekday'] = pd.DatetimeIndex(data['Date']).weekday
    data['quarter'] = pd.DatetimeIndex(data['Date']).quarter
    data['is_month_start'] = pd.DatetimeIndex(data['Date']).is_month_start
    data['is_month_end'] = pd.DatetimeIndex(data['Date']).is_month_end

    data = data.drop(['Date','activated_at_local','activated_at','obfuscated_item_names'], axis = 1) 
    data.fillna("", inplace=True)

    return data

def split_train_test(data, split_perc):
    print("data train test split started")
    size = len(data)
    print('feature size %s', str(len(data.columns)))
    data_train = data[0:int(split_perc * size)]
    data_test = data[int(split_perc * size):size]
    return data_train,data_test

def train_model(data_train, data_test):
    print("model training started")
    xtr, xts = data_train.drop(['prep_time_seconds'], axis=1), data_test.drop(['prep_time_seconds'], axis=1)
    ytr, yts = data_train['prep_time_seconds'].values, data_test['prep_time_seconds'].values

    categorical_var = np.where((xtr.dtypes != np.float) & (xtr.dtypes != np.int))[0]

    #hyper param optimization nee to add
    model = RandomForestRegressor(n_estimators=200,max_depth=8,min_samples_split=2, min_samples_leaf=1, 
                              criterion='mse',
                              max_features='sqrt',
                              random_state=42,
                              verbose=1,
                              n_jobs=2)
    model.fit(xtr, ytr)
    y_pred = model.predict(xts)
    rmse_error = np.sqrt(MSE(yts, y_pred))
    mae_error = MAE(yts, y_pred)
    
    print('rmse %.5f, mae %.5f ' % (rmse_error, mae_error))
    return model,rmse_error,mae_error


# In[3]:


from sklearn import preprocessing
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.ensemble import RandomForestRegressor
from hyperopt import tpe,hp,Trials
from hyperopt.fmin import fmin
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import os
import datetime
import numpy as np
data = pd.read_json('orders.json')
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE


split_perc = 0.8
feature_columns = ['import_source','activated_at','activated_at_local','kitchen_id','obfuscated_item_names','subtotal','prep_time_seconds']


#dml.logit({"train":"started data prep"})

#dml.logit({"status":"started"})
data = pd.DataFrame()
data = read_json_data("orders.json")

data = feature_engg(data, feature_columns)

cats = []
for col in data.columns.values:
    if data[col].dtype == 'object':
        cats.append(col)

df_cont = data.drop(cats, axis=1)
df_cat = data[cats]

#dml.logit({"message":"missing_value_adjustment in numerical features"})

for col in df_cont.columns.values:
    if np.sum(df_cont[col].isnull()) > 10:
        df_cont = df_cont.drop(col, axis = 1)
    elif np.sum(df_cont[col].isnull()) > 0:
        median = df_cont[col].median()
        idx = np.where(df_cont[col].isnull())[0]
        df_cont[col].iloc[idx] = median

        outliers = np.where(is_outlier(df_cont[col]))
        df_cont[col].iloc[outliers] = median

        if skew(df_cont[col]) > 0.9:
            df_cont[col] = np.log(df_cont[col])
            df_cont[col] = df_cont[col].apply(lambda x: 0 if x == -np.inf else x)

        df_cont[col] = Normalizer().fit_transform(df_cont[col].reshape(1,-1))[0]

#dml.logit({"message":"missing_value_adjustment in categorical features"})

for col in df_cat.columns.values:
    if np.sum(df_cat[col].isnull()) > 10:
        df_cat = df_cat.drop(col, axis = 1)
        continue
    elif np.sum(df_cat[col].isnull()) > 0:
        df_cat[col] = df_cat[col].fillna('-1')
    le = preprocessing.LabelEncoder()
    df_cat[col] = le.fit_transform(df_cat[col])

    num_cols = df_cat[col].max()
    for i in range(num_cols):
        col_name = col + '_' + str(i)
        df_cat[col_name] = df_cat[col].apply(lambda x: 1 if x == i else 0)

#dml.logit({"message":"join df_cont df_cat "})

data_transformed = df_cont.join(df_cat)

data_train,data_test = split_train_test(data_transformed, split_perc)
# model, rmse_error,mae_error = train_model(data_train,data_test)

# print("RF Trained model saved.")

# dml.logit({"metric_type":"test", "metrics":{"rmse":rmse_error,"mae":mae_error}})#metrics

# dml.logit({"status":"complete"})


# # data collection 
# - normalization
# - scaling
# - skewness
# - categorigization
# - encoding techniques
# 
# 
# # model exploration phase
# 
# - model build base version for benchmark
# - iterations of various types of model, tree based, deepnext, various architectures, stacking, ensemble
# - hyperparam tuning within each model
# - additional feature engg 
# 
# # peer and stackholder review
# - show charts, benhcmarks
# - data drift/anomalies
# - understand effect of each trial, ignore poor models, select best ones
# - show best ones on ppt, auto report to start with
# - track it systematically, share what worked what failed
# - decide as a team and push models to prod
# 

# In[4]:


get_ipython().system(' pip install /Users/jyotirmoysundi/git/distml_logger/dist/chaya_ai-0.0.1.tar.gz')

#/Users/jyotirmoysundi/git/distml_logger/dist/chaya_ai-0.0.1.tar.gz


# In[20]:



from chaya_ai.chaya_ai import tracker 

cai = tracker() # increase collaboration and precise feedback

cai.setup(config="/Users/jyotirmoysundi/Downloads/distml.json", project_name="temp_pred",           track={"start_tag":{"keywords":"start|train"},"end_tag":{"metrics":"rmse|mae"}})


# In[ ]:





# In[25]:


import lightgbm as lgb
from catboost import CatBoostRegressor

cai.model_name = "ensemble_stacked_model_4_models"

cai.save({"train":"start"})

cai.detect_drift(data_train,data_test) # drift in features

xtr, xts = data_train.drop(['prep_time_seconds'], axis=1), data_test.drop(['prep_time_seconds'], axis=1)
ytr, yts = data_train['prep_time_seconds'].values, data_test['prep_time_seconds'].values

categorical_var = np.where((xtr.dtypes != np.float) & (xtr.dtypes != np.int))[0]

lgtrain = lgb.Dataset(xtr, ytr, categorical_feature = ['is_month_start','is_month_end'])
lgvalid = lgb.Dataset(xts, yts, categorical_feature = ['is_month_start','is_month_end'])


params = {
    'objective': 'quantile',
    'metric': 'quantile',
    'num_leaves' : 100,
    'max_depth': 5,
    'learning_rate' : 0.01,
    'feature_fraction' : 0.6,
    'verbosity' : -1,
    'is_training_metric': True,
    'n_estimators' : 500
}

model = lgb.LGBMRegressor(**params, alpha= 0.9)
model.fit(xtr, ytr, eval_set=[(xts, yts), (xtr, ytr)], verbose=10)

y_pred_lgbm = model.predict(xts)

model_cb = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.01, loss_function='Quantile:alpha=0.95')

model_cb.fit(xtr, ytr, cat_features = categorical_var, plot=False)

y_pred_catboost = model_cb.predict(xts)

y_pred = (0.1*y_pred_lgbm + 0.9*y_pred_catboost)

# model_rf = RandomForestRegressor(n_estimators=1000,max_depth=4, 
#                           random_state=42,
#                           max_features ='sqrt',
#                           n_jobs=2,
#                           verbose=1)
# model_rf.fit(xtr, ytr)
# y_pred = model_rf.predict(xts)

rmse_error = np.sqrt(MSE(yts, y_pred))
mae_error = MAE(yts, y_pred)
cai.save({"metric_type":"test", "metrics":{"rmse":rmse_error,"mae":mae_error}})#metrics

fig, ax = plt.subplots(figsize=(10, 7))
lgb.plot_importance(model, max_num_features=30, ax=ax)
plt.title("LightGBM - Feature Importance _qr_0.95 ");
plt.savefig('feature_importance_qr_0.95.png')

cai.saveplot("feature_importance_qr_0.95.png")


# In[26]:


model_cb.save_model("model_cb")
model.booster_.save_model('lgbmmodel')


# In[28]:


cai.save_model("model_cb")
cai.save_model("lgbmmodel")


# In[ ]:


from google.cloud import storage

def upload_to_bucket(blob_name, path_to_file, bucket_name):
    """ Upload data to a bucket"""

    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        'creds.json')

    #print(buckets = list(storage_client.list_buckets())

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)

    #returns a public url
    return blob.public_url


# In[30]:


dml.model_name = "random_forest"
dml.logit({"train":"start"})

#dml.detect_drift(data_train,data_test) # drift in features

xtr, xts = data_train.drop(['prep_time_seconds'], axis=1), data_test.drop(['prep_time_seconds'], axis=1)
ytr, yts = data_train['prep_time_seconds'].values, data_test['prep_time_seconds'].values

categorical_var = np.where((xtr.dtypes != np.float) & (xtr.dtypes != np.int))[0]

#hyper param optimization nee to add
model = RandomForestRegressor(n_estimators=1000,max_depth=4, 
                          random_state=42,
                          max_features ='sqrt',
                          n_jobs=2,
                          verbose=1)
model.fit(xtr, ytr)
y_pred = model.predict(xts)
rmse_error = np.sqrt(MSE(yts, y_pred))
mae_error = MAE(yts, y_pred)
dml.logit({"metric_type":"test", "metrics":{"rmse":rmse_error,"mae":mae_error}})#metrics

# fig, ax = plt.subplots(figsize=(10, 7))
# lgb.plot_importance(lgb_clf, max_num_features=30, ax=ax)
# plt.title("LightGBM - Feature Importance _qr_0.95 ");
# plt.savefig('feature_importance_qr_0.95.png')

# dml.saveplot("feature_importance_qr_0.95.png")


# In[19]:


#! pip3 uninstall distml-logger-0.0.1
#! pip3 install --force-reinstall /Users/jyotirmoysundi/git/distml_logger/dist/distml_logger-0.0.1.tar.gz 


# In[34]:


#%load_ext autoreload
get_ipython().run_line_magic('autosave', '1')


# In[16]:





# In[321]:


(y_pred + y_pred)/2


# In[98]:


lgb.LGBMModel.best_score_


# In[31]:


fig, ax = plt.subplots(figsize=(10, 7))
lgb.plot_importance(lgb_clf, max_num_features=30, ax=ax)
plt.title("LightGBM - Feature Importance");
plt.savefig('feature_importance.png')

dml.saveplot("feature_importance.png")


# In[314]:


import seaborn as sns

dml.model_name = "catboost"

dml.logit({"train":"start"})


dml.detect_drift(data_train,data_test) # drift in features

xtr, xts = data_train.drop(['prep_time_seconds'], axis=1), data_test.drop(['prep_time_seconds'], axis=1)
ytr, yts = data_train['prep_time_seconds'].values, data_test['prep_time_seconds'].values

categorical_var = np.where(xtr.dtypes != np.float)[0]

#model = CatBoostRegressor(iterations=50, depth=6, learning_rate=0.01, loss_function='RMSE')
model = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.01, loss_function='Quantile:alpha=0.55')

#tuned hyper params iterations, depth,lr, manually, ideally would use hyperopt [easy to use hyperopt for bayesion hyper params tunes]

dml.logit({"msg":"started model fit.."})#metrics

model.fit(xtr, ytr, cat_features = categorical_var, plot=False)

dml.logit({"msg":"model fit complete.."})#metrics

y_pred = model.predict(xts)

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

rmse_error = np.sqrt(MSE(yts, y_pred))
mae_error = MAE(yts, y_pred)

# sns_plot = sns.distplot(np.log(data['prep_time_seconds']),kde=False)
# sns_plot.figure.savefig("output_chart.png")
# dml.saveplot("output_chart.png")

#dml.logit({"metric_type":"train", "metrics":{"rmse":rmse_error,"mae":mae_error}})#metrics
dml.logit({"metric_type":"test", "metrics":{"rmse":rmse_error,"mae":mae_error}})#metrics

print('rmse %.5f, mae %.5f ' % (rmse_error, mae_error))
dml.logit({"status":"complete"})

data = pd.DataFrame({'feature_importance': model.get_feature_importance(), 
              'feature_names': xtr.columns}).sort_values(by=['feature_importance'], 
                                                       ascending=False)
ax = data[:30].sort_values(by=['feature_importance'], ascending=True).plot.barh(x='feature_names', y='feature_importance')
plt.title('catboost_feature_importances Feature Importances')  
plot_name = "catboost_feature_importances.png"
plt.savefig(plot_name)
dml.saveplot(plot_name)
shap.force_plot(explainer.expected_value, shap_values[0,:], xtr.iloc[0,:], matplotlib=True, show=False).savefig('shap_summary.png')
plt.title('shap variable importance '+ str('Quantile:alpha=0.4'))  
dml.saveplot("shap_summary.png")



# In[299]:


# res = model.calc_feature_statistics(xtr,
#                                     ytr,
#                                     feature=3,
#                                     plot=True)

# plt.title('catboost_feature_importances Feature Importances')  
# #plot_name = "catboost_feature_importances.png"
# #plt.savefig(plot_name)
# #dml.saveplot(plot_name)


# In[310]:


from catboost import CatBoostClassifier, Pool
import shap

shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Pool(xtr, ytr, cat_features=categorical_var))
shap.summary_plot(shap_values, xtr, plot_type="bar")


# In[298]:


shap.force_plot(explainer.expected_value, shap_values[0,:], xtr.iloc[0,:], matplotlib=True, show=False).savefig('shap_summary.png')
#shap.save_html("shap_summary.html", f)
plt.title('shap variable importance')  

dml.saveplot("shap_summary.png")


# In[58]:


xts.day.plot()


# In[138]:


import matplotlib.pyplot as plt # Impot the relevant module

fig, ax = plt.subplots() # Create the figure and axes object

# Plot the first x and y axes:
xtr.day.plot(label='train')
plt.legend( bbox_to_anchor=(0.1,0.92), loc=10)
xts.day.plot(ax = ax, secondary_y = True, label='test')
plt.legend( bbox_to_anchor=(0.7,0.94), loc=10)

ax.set(title = "Data Distribution for feature Day",
       xlabel = "Data Index ",
       ylabel = "Data Values"
      )

plt.title("Data Distribution for feature Day");
plt.savefig('feature_importance_day_pname.png')

# Plot the second x and y axes. By secondary_y = True a second y-axis is requested:
# (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html for details)


# In[139]:


import matplotlib.pyplot as plt # Impot the relevant module

fig, ax = plt.subplots() # Create the figure and axes object

# Plot the first x and y axes:
xtr.dayofyear.plot(label='train')
plt.legend( bbox_to_anchor=(0.1,0.92), loc=10)
xts.dayofyear.plot(ax = ax, secondary_y = True, label='test')
plt.legend( bbox_to_anchor=(0.7,0.94), loc=10)

ax.set(title = "Data Distribution for feature Day",
       xlabel = "Data Index ",
       ylabel = "Data Values"
      )

plt.title("Data Distribution for feature Day");
plt.savefig('feature_importance_day_pname.png')

# Plot the second x and y axes. By secondary_y = True a second y-axis is requested:
# (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html for details)


# In[166]:


xtr.columns


# In[178]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = 16,12

sns.set_theme(style="darkgrid")
xtr["dataset_type"] = "train"
xts["dataset_type"] = "test"
data_concat = pd.concat([xtr, xts])
plt.clf()
if len(data_concat["subtotal"].unique()) > 20:
    
    #data_concat['subtotal_qrank'] = pd.qcut(data_concat['subtotal'], 10, labels = False)
    data_concat['subtotal_rank'] = data_concat['subtotal'].rank(method='first')
    data_concat['subtotal_qrank'] = pd.qcut(data_concat['subtotal_rank'].values, 20).codes


    plot = sns.countplot(x="subtotal_qrank", hue="dataset_type", data=data_concat)
    plt.setp(plot.get_xticklabels(), rotation=90)
    plt.title("Data Distribution for feature dayofyear");
    plt.savefig('feature_importance_dayofyear_pname.png')
else:
    plot = sns.countplot(x="subtotal", hue="dataset_type", data=data_concat)
    plt.setp(plot.get_xticklabels(), rotation=90)
    plt.title("Data Distribution for feature dayofyear");
    plt.savefig('feature_importance_dayofyear_pname.png')


# In[173]:


len(data_concat["subtotal"].unique())

