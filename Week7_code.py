import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("graphs_week7",exist_ok=True)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib

sns.set_style("whitegrid")


df=pd.read_csv("crop_yield.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

for col in ['Fertilizer','Pesticide','Area']:
    df[col]=np.log1p(df[col])

df['Fertilizer_per_Area']=df['Fertilizer']/(df['Area']+1e-5)
df['Pesticide_per_Area']=df['Pesticide']/(df['Area']+1e-5)
df['Rainfall_per_Area']=df['Annual_Rainfall']/(df['Area']+1e-5)
df['Input_Intensity']=df['Fertilizer']+df['Pesticide']

x=df[['State','Crop','Season','Crop_Year','Area','Annual_Rainfall','Fertilizer','Pesticide',
      'Fertilizer_per_Area','Pesticide_per_Area','Rainfall_per_Area','Input_Intensity']]
y=df['Yield']

preprocessor=ColumnTransformer([
    ('num',StandardScaler(),['Crop_Year','Area','Annual_Rainfall','Fertilizer','Pesticide',
                             'Fertilizer_per_Area','Pesticide_per_Area','Rainfall_per_Area','Input_Intensity']),
    ('cat',OneHotEncoder(drop='first'),['State','Crop','Season'])
])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


rf=Pipeline([('preprocessor',preprocessor),
             ('model',RandomForestRegressor(random_state=42))])

rf.fit(x_train,y_train)

gb=Pipeline([('preprocessor',preprocessor),
             ('model',GradientBoostingRegressor(random_state=42))])

gb.fit(x_train,y_train)

rf_score=r2_score(y_test,rf.predict(x_test))
gb_score=r2_score(y_test,gb.predict(x_test))

# WEEK 7


final_model = rf if rf_score>gb_score else gb

print("Final Model Selected")


model = final_model.named_steps['model']
importances = model.feature_importances_

plt.bar(range(len(importances)), importances)
plt.savefig("graphs_week7/feature_importance.png")
plt.close()


errors = y_test - final_model.predict(x_test)
sns.histplot(errors)
plt.savefig("graphs_week7/error_distribution.png")
plt.close()

joblib.dump(final_model,"final_model.pkl")

print("Model saved!")

def predict_yield(data):
    return final_model.predict(pd.DataFrame([data]))[0]