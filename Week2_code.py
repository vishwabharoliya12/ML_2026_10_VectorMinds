import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg') 
import os
os.makedirs("graphs", exist_ok=True) 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")


df = pd.read_csv("/workspaces/ML_2026_10_VectorMinds/crop_yield.csv")

print("Shape:", df.shape)
print(df.head())
print(df.info())

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("Missing Values:\n",df.isnull().sum())

# EDA SECTION

numerical_features = [
    'Crop_Year','Area',
    'Annual_Rainfall',
    'Fertilizer','Pesticide',
    'Yield'
]

categorical_features=['State', 'Crop', 'Season']

# Distribution Plots

for feature in numerical_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.savefig(f"{feature}_distribution.png")
    plt.close()

# Boxplots (Outliers)

for feature in numerical_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot of {feature}")
    plt.savefig(f"{feature}_boxplot.png")
    plt.close()

# Correlation Heatmap

plt.figure(figsize=(8,6))
sns.heatmap(df[numerical_features + ['Yield']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.close()

# Scatter Plots vs Yield

for feature in ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[feature], y=df['Yield'])
    plt.title(f"{feature} vs Yield")
    plt.savefig(f"{feature}_vs_Yield.png")
    plt.close()


#  Categorical Analysis

for feature in categorical_features:
    plt.figure(figsize=(10,5))
    sns.boxplot(x=feature, y='Yield', data=df)
    plt.xticks(rotation=45)
    plt.title(f"{feature} vs Yield")
    plt.savefig(f"{feature}_vs_Yield_boxplot.png")
    plt.close()



X = df[['State','Crop','Season',
        'Crop_Year', 'Area',
        'Annual_Rainfall',
        'Fertilizer', 'Pesticide']]

y = df['Yield']

preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(), 
         ['Crop_Year', 'Area',
          'Annual_Rainfall',
          'Fertilizer', 'Pesticide']),
        ('cat', OneHotEncoder(drop='first'),
         ['State', 'Crop', 'Season'])
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


linear_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

linear_pipeline.fit(X_train, y_train)
y_pred_lr = linear_pipeline.predict(X_test)

# Ridge Regression

ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])

ridge_pipeline.fit(X_train, y_train)
y_pred_ridge = ridge_pipeline.predict(X_test)

# Lasso Regression

lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Lasso(alpha=0.1))
])

lasso_pipeline.fit(X_train, y_train)
y_pred_lasso = lasso_pipeline.predict(X_test)


def evaluate_model(y_test, y_pred, name):
    print(f"\n{name} Performance")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))

evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
evaluate_model(y_test, y_pred_lasso, "Lasso Regression")