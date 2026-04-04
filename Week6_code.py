import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("graphs_week5",exist_ok=True)
os.makedirs("graphs_week6",exist_ok=True)

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

sns.set_style("whitegrid")

df=pd.read_csv("crop_yield.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

num_cols=['Area','Annual_Rainfall','Fertilizer','Pesticide','Yield']

for col in ['Fertilizer','Pesticide','Area']:
    df[col]=np.log1p(df[col])

for col in num_cols:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    df[col]=df[col].clip(Q1-1.5*IQR,Q3+1.5*IQR)

df['Fertilizer_per_Area']=df['Fertilizer']/(df['Area']+1e-5)
df['Pesticide_per_Area']=df['Pesticide']/(df['Area']+1e-5)
df['Rainfall_per_Area']=df['Annual_Rainfall']/(df['Area']+1e-5)
df['Input_Intensity']=df['Fertilizer']+df['Pesticide']

x=df[['State','Crop','Season','Crop_Year','Area',
        'Annual_Rainfall','Fertilizer','Pesticide',
        'Fertilizer_per_Area','Pesticide_per_Area',
        'Rainfall_per_Area','Input_Intensity']]

y=df['Yield']

numerical_features=['Crop_Year','Area','Annual_Rainfall',
                      'Fertilizer','Pesticide',
                      'Fertilizer_per_Area','Pesticide_per_Area',
                      'Rainfall_per_Area','Input_Intensity']

categorical_features=['State','Crop','Season']

preprocessor = ColumnTransformer([
    ('num',StandardScaler(),numerical_features),
    ('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),categorical_features)
])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


#TUNING


rf_pipeline=Pipeline([
    ('preprocessor',preprocessor),
    ('model',RandomForestRegressor(random_state=42))
])

rf_params={
    'model__n_estimators':[100,200],
    'model__max_depth':[None,10],
}

rf_grid=GridSearchCV(rf_pipeline,rf_params,cv=5,scoring='r2',n_jobs=-1)
rf_grid.fit(x_train,y_train)

gb_pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',GradientBoostingRegressor(random_state=42))
])

gb_params={
    'model__n_estimators':[100,200],
    'model__learning_rate':[0.05,0.1],
}

gb_grid=GridSearchCV(gb_pipeline, gb_params, cv=5, scoring='r2', n_jobs=-1)
gb_grid.fit(x_train,y_train)

best_rf=rf_grid.best_estimator_
best_gb=gb_grid.best_estimator_

rf_preds=best_rf.predict(x_test)
gb_preds=best_gb.predict(x_test)

print("RF R2:",r2_score(y_test,rf_preds))
print("GB R2:",r2_score(y_test,gb_preds))

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    best_rf,x,y,cv=5,scoring='r2',
    train_sizes=np.linspace(0.1,1.0,5),n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1))
plt.plot(train_sizes, test_scores.mean(axis=1))
plt.savefig("graphs_week6/learning_curve.png")
plt.close()