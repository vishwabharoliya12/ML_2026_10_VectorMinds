import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


os.makedirs("graphs_mid", exist_ok=True)

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score, KFold)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               StackingRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint, uniform

sns.set_style("whitegrid")


df = pd.read_csv("/workspaces/ML_2026_10_VectorMinds/crop_yield.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("Dataset shape:",df.shape)

x = df[['State','Crop','Season',
        'Crop_Year','Area',
        'Annual_Rainfall','Fertilizer','Pesticide']]

y = df['Yield']

numerical_features = [
    'Crop_Year','Area','Annual_Rainfall',
    'Fertilizer','Pesticide'
]

categorical_features = ['State','Crop','Season']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

x_train,x_test,y_train,y_test =train_test_split(
    x,y,test_size=0.2, random_state=42
)

def evaluate(name, y_true, y_pred):
    mae= mean_absolute_error(y_true, y_pred)
    rmse= np.sqrt(mean_squared_error(y_true, y_pred))
    r2= r2_score(y_true, y_pred)

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


rf_base_pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

rf_base_pipeline.fit(x_train, y_train)

base_preds=rf_base_pipeline.predict(x_test)

base_result = evaluate(
    "RF Baseline (Week 3)",
    y_test,
    base_preds
)

print("\n>>> Hyperparameter Tuning: Random Forest")

param_dist_rf ={
    'model__n_estimators': randint(100, 400),
    'model__max_depth': [None, 10, 15, 20, 25],
    'model__min_samples_split': randint(2, 20),
    'model__min_samples_leaf': randint(1, 10),
    'model__max_features': ['sqrt', 'log2', 0.5]
}

rf_pipeline=Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ))
])

rf_random_search=RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_dist_rf,
    n_iter=30,
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_random_search.fit(x_train, y_train)
best_rf = rf_random_search.best_estimator_
rf_tuned_preds = best_rf.predict(x_test)

rf_tuned_result = evaluate(
    "Random Forest (Tuned)",
    y_test,
    rf_tuned_preds
)

print("\nBest RF Params:", rf_random_search.best_params_)

print("\n>>> Hyperparameter Tuning: Gradient Boosting")

param_grid_gb = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],
    'model__subsample': [0.8, 1.0]
}

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

gb_grid_search = GridSearchCV(
    gb_pipeline,
    param_grid=param_grid_gb,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

gb_grid_search.fit(x_train, y_train)
best_gb =gb_grid_search.best_estimator_
gb_tuned_preds =best_gb.predict(x_test)

gb_tuned_result =evaluate(
    "Gradient Boosting (Tuned)",
    y_test,
    gb_tuned_preds
)

print("\nBest GB Params:",gb_grid_search.best_params_)


print("\n>>> Building Stacking Ensemble")

x_train_proc =preprocessor.fit_transform(x_train)
x_test_proc =preprocessor.transform(x_test)

base_estimators = [
    ('rf', RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )),

    ('gb', GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )),

    ('ridge', Ridge(alpha=1.0))
]

meta_learner=Ridge(alpha=0.5)

stacking_model =StackingRegressor(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

stacking_model.fit(x_train_proc, y_train)

stacking_preds = stacking_model.predict(x_test_proc)

stacking_result = evaluate(
    "Stacking Ensemble",
    y_test,
    stacking_preds
)


print("\n===== Final 5-Fold Cross Validation =====")

kf =KFold(n_splits=5, shuffle=True, random_state=42)

cv_models ={
    "RF Baseline": rf_base_pipeline,
    "RF Tuned": best_rf,
    "GB Tuned": best_gb
}

for name, model in cv_models.items():

    scores = cross_val_score(
        model,
        x,
        y,
        cv=kf,
        scoring='r2',
        n_jobs=-1
    )

    print(
        f"{name:25s}: "
        f"Mean R² = {scores.mean():.4f} ± {scores.std():.4f}"
    )


from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    best_rf,
    x,
    y,
    cv=3,
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1
)

train_mean =train_scores.mean(axis=1)
val_mean= val_scores.mean(axis=1)

plt.figure(figsize=(8,5))

plt.plot(train_sizes, train_mean, 'o-', label="Training R²")
plt.plot(train_sizes, val_mean, 's-', label="Validation R²")

plt.xlabel("Training Size")
plt.ylabel("R² Score")
plt.title("Learning Curves — Tuned Random Forest")
plt.legend()

plt.tight_layout()
plt.savefig("graphs_mid/learning_curves_rf.png", dpi=150)
plt.close()


all_results = [
    base_result,
    rf_tuned_result,
    gb_tuned_result,
    stacking_result
]

final_df = pd.DataFrame(all_results)

plt.figure(figsize=(10,5))

sns.barplot(
    data=final_df,
    x="Model",
    y="R2"
)

plt.xticks(rotation=20)
plt.title("Week 4 Model Comparison")

plt.tight_layout()
plt.savefig("graphs_mid/final_model_comparison_week4.png", dpi=150)
plt.close()


rf_tuned_model = best_rf.named_steps['model']

ohe = best_rf.named_steps['preprocessor']\
    .named_transformers_['cat']

cat_cols =ohe.get_feature_names_out(categorical_features)

all_features =numerical_features + list(cat_cols)

importances =rf_tuned_model.feature_importances_

feat_df=pd.DataFrame({
    "Feature": all_features[:len(importances)],
    "Importance": importances
}).sort_values("Importance", ascending=False).head(20)

plt.figure(figsize=(10,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feat_df
)

plt.title("Top 20 Feature Importances")

plt.tight_layout()
plt.savefig("graphs_mid/feature_importance_tuned_rf.png", dpi=150)
plt.close()

best_preds =rf_tuned_preds
plt.figure(figsize=(7,6))

plt.scatter(y_test, best_preds, alpha=0.3)

lims = [
    min(y_test.min(), best_preds.min()),
    max(y_test.max(), best_preds.max())
]

plt.plot(lims, lims, 'r--')

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")

plt.title("Actual vs Predicted — Tuned RF")

plt.tight_layout()
plt.savefig("graphs_mid/best_model_actual_vs_predicted.png", dpi=150)
plt.close()

errors = y_test.values - best_preds

plt.figure(figsize=(8,4))

sns.histplot(errors, kde=True)

plt.axvline(0, color='red', linestyle='--')

plt.title("Prediction Error Distribution")

plt.tight_layout()
plt.savefig("graphs_mid/prediction_error_distribution.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────
print("\n========== WEEK 4 FINAL SUMMARY ==========")

print(
    final_df
    .set_index("Model")
    .round(4)
)

print("\nBest Model by R²:",
      final_df.loc[final_df['R2'].idxmax(), 'Model'])

print("\nGraphs saved in: graphs_mid/")