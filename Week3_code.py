import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


os.makedirs("graphs", exist_ok=True)

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")


df = pd.read_csv("/workspaces/ML_2026_10_VectorMinds/crop_yield.csv")

print("Shape:",df.shape)
print(df.head())
print(df.info())

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("Missing Values:\n",df.isnull().sum())


X = df[['State','Crop','Season',
        'Crop_Year','Area',
        'Annual_Rainfall',
        'Fertilizer','Pesticide']]

y = df['Yield']

numerical_features = ['Crop_Year','Area',
                      'Annual_Rainfall',
                      'Fertilizer','Pesticide']

categorical_features = ['State','Crop','Season']


# PREPROCESSING

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# MODEL PIPELINES

dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42))
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, max_depth=15,
                                    min_samples_split=10, random_state=42))
])

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=100,
                                        learning_rate=0.1,
                                        max_depth=5,
                                        random_state=42))
])

svr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
])


# MODEL TRAINING

dt_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)
gb_pipeline.fit(X_train, y_train)
svr_pipeline.fit(X_train, y_train)

y_pred_dt = dt_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)
y_pred_gb = gb_pipeline.predict(X_test)
y_pred_svr = svr_pipeline.predict(X_test)


# MODEL EVALUATION

def evaluate_model(y_test, y_pred, name):
    print(f"\n{name} Performance")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))


evaluate_model(y_test, y_pred_dt, "Decision Tree")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")
evaluate_model(y_test, y_pred_svr, "Support Vector Regression")


# CROSS VALIDATION

print("\nCross Validation (R2 Scores)")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in [
    ("Decision Tree", dt_pipeline),
    ("Random Forest", rf_pipeline),
    ("Gradient Boosting", gb_pipeline)
]:
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    print(f"{name} Mean R2:", scores.mean())


# FEATURE IMPORTANCE (Random Forest)

rf_model = rf_pipeline.named_steps['model']

ohe = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_names = ohe.get_feature_names_out(categorical_features)

all_features = numerical_features + list(cat_names)

importance = rf_model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": importance
}).sort_values("Importance", ascending=False).head(20)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Random Forest Feature Importance")
plt.savefig("rf_feature_importance.png")
plt.close()


# ACTUAL VS PREDICTED PLOT

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Random Forest: Actual vs Predicted")
plt.savefig("rf_actual_vs_predicted.png")
plt.close()


# RESIDUAL PLOT

residuals = y_test - y_pred_rf

plt.figure(figsize=(6,6))
plt.scatter(y_pred_rf, residuals, alpha=0.3)
plt.axhline(0, linestyle='--')
plt.xlabel("Predicted Yield")
plt.ylabel("Residual")
plt.title("Residual Plot (Random Forest)")
plt.savefig("rf_residual_plot.png")
plt.close()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_gb, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Gradient Boosting: Actual vs Predicted")
plt.savefig("gb_actual_vs_predicted.png")
plt.close()
# MODEL COMPARISON

results = {
    "Decision Tree": r2_score(y_test, y_pred_dt),
    "Random Forest": r2_score(y_test, y_pred_rf),
    "Gradient Boosting": r2_score(y_test, y_pred_gb),
    "SVR": r2_score(y_test, y_pred_svr)
}

plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values())
plt.ylabel("R2 Score")
plt.title("Model Comparison")
plt.savefig("week3_model_comparison.png")
plt.close()


# DECISION TREE VISUALIZATION

dt_model = dt_pipeline.named_steps['model']

X_transformed = dt_pipeline.named_steps['preprocessor'].fit_transform(X_train)

plt.figure(figsize=(20,6))
plot_tree(dt_model,
          max_depth=3,
          filled=True,
          fontsize=8)
plt.title("Decision Tree Structure")
plt.savefig("decision_tree_structure.png")
plt.close()