import numpy as np
import pandas as pd
import seaborn as sns
import os
import warnings
import joblib
import optuna
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
os.makedirs("graphs_week8",exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sns.set_style("whitegrid","ticks")
seed=42

df=pd.read_csv("crop_yield.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

for c in ['Fertilizer','Pesticide','Area']:
    capping=df[c].quantile(0.99)
    df[c]=df[c].clip(upper=capping)
    df[c]=np.log1p(df[c])

df['Fertilizer_per_Area']=df['Fertilizer']/(df['Area']+1e-5)
df['Pesticide_per_Area']=df['Pesticide']/(df['Area']+1e-5)
df['Input_Intensity']=df['Fertilizer']+df['Pesticide']

y_log=np.log1p(df['Yield'])

feat_cols=['State','Crop','Season','Crop_Year','Area','Annual_Rainfall','Fertilizer','Pesticide',
                'Fertilizer_per_Area','Pesticide_per_Area','Input_Intensity']
num_cols=['Crop_Year','Area','Annual_Rainfall','Fertilizer','Pesticide',
            'Fertilizer_per_Area', 'Pesticide_per_Area', 'Input_Intensity']
cat_cols=['State','Crop','Season']

x=df[feat_cols]

x_train,x_test,y_train_log,y_test_log=train_test_split(x, y_log,test_size=0.2,random_state=seed)
y_test_raw=np.expm1(y_test_log) 
print(f"Train: {len(x_train)}|Test: {len(x_test)}")



prepro=ColumnTransformer([
    ('n',StandardScaler(), num_cols),
    ('cat',OneHotEncoder(drop='first',handle_unknown='ignore',
                          sparse_output=False),cat_cols)
])

X_train_proc=prepro.fit_transform(x_train)
X_test_proc=prepro.transform(x_test)
ohe_names=prepro.named_transformers_['cat']\
                                 .get_feature_names_out(cat_cols)
all_feature_names=np.array(num_cols + list(ohe_names))
print(f"Total features after encoding: {X_train_proc.shape[1]}")


def evaluate(name, y_true_raw, y_pred_log):
    y_pred_raw=np.expm1(y_pred_log)
    mae=mean_absolute_error(y_true_raw, y_pred_raw)
    rmse=np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    r2=r2_score(y_true_raw, y_pred_raw)
    r2_log=r2_score(np.log1p(y_true_raw), y_pred_log)
    print(f"\n{name}")
    print(f"MAE(raw):{mae:.2f}")
    print(f"RMSE(raw):{rmse:.2f}")
    print(f"R2(raw):{r2:.4f}")
    print(f"R2(log):{r2_log:.4f}")
    return {"Model": name, "MAE": round(mae, 2),
            "RMSE": round(rmse, 2), "R2_raw": round(r2, 4),
            "R2_log": round(r2_log, 4)}


print("\n------------ BASELINE MODELS------------")

rf=RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
rf.fit(X_train_proc, y_train_log)
res_rf=evaluate("Random Forest (baseline)", y_test_raw, rf.predict(X_test_proc))

gb=GradientBoostingRegressor(n_estimators=50, random_state=seed)
gb.fit(X_train_proc, y_train_log)
res_gb=evaluate("Gradient Boosting (baseline)", y_test_raw, gb.predict(X_test_proc))

xgb=XGBRegressor(n_estimators=50, random_state=seed, verbosity=0, n_jobs=-1)
xgb.fit(X_train_proc, y_train_log)
res_xgb=evaluate("XGBoost (baseline)", y_test_raw, xgb.predict(X_test_proc))


print("\n---OPTUNA: RANDOM FOREST--- ")

def rf_objective(trial):
    model=RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 50, 150),
        max_depth=trial.suggest_int("max_depth", 5, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        max_features=trial.suggest_categorical("max_features",
                                                       ["sqrt", "log2"]),
        random_state=seed,n_jobs=-1
    )
    return cross_val_score(model, X_train_proc, y_train_log,
                           cv=2, scoring='r2', n_jobs=-1).mean()

rf_study=optuna.create_study(direction='maximize')
rf_study.optimize(rf_objective, n_trials=8)
print(f"Best RF params : {rf_study.best_params}")

best_rf=RandomForestRegressor(**rf_study.best_params, random_state=seed, n_jobs=-1)
best_rf.fit(X_train_proc, y_train_log)
res_rf_tuned=evaluate("RF Tuned", y_test_raw, best_rf.predict(X_test_proc))


print("\n--- OPTUNA: XGBOOST ---")

def xgb_objective(trial):
    model= XGBRegressor(
        n_est= trial.suggest_int("n_estimators", 50, 150),
        max_depth= trial.suggest_int("max_depth", 3, 8),
        l_rate= trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        subsample= trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree= trial.suggest_float("colsample_bytree", 0.7, 1.0),
        random_state=seed,verbosity=0, n_jobs=-1
    )
    return cross_val_score(model, X_train_proc, y_train_log,
                           cv=2, scoring='r2', n_jobs=-1).mean()

xgb_study=optuna.create_study(direction='maximize')
xgb_study.optimize(xgb_objective, n_trials=8)
print(f"Best XGB params: {xgb_study.best_params}")

best_xgb=XGBRegressor(**xgb_study.best_params, random_state=seed,
                         verbosity=0, n_jobs=-1)
best_xgb.fit(X_train_proc, y_train_log)
res_xgb_tuned=evaluate("XGBoost Tuned", y_test_raw, best_xgb.predict(X_test_proc))


print("\n----STACKING ENSEMBLE -----")

base_learners=[
    ('rf',RandomForestRegressor(**rf_study.best_params,random_state=seed,n_jobs=-1)),
    ('xgb',XGBRegressor(**xgb_study.best_params,random_state=seed,verbosity=0,n_jobs=-1)),
    ('gb',  GradientBoostingRegressor(n_estimators=50,random_state=seed))
]

stack=StackingRegressor(
    estimators=base_learners,
    final_estimator=Ridge(alpha=1.0),
    cv=3,           
    n_jobs=-1
)

stack.fit(X_train_proc, y_train_log)
res_stack=evaluate("Stacking Ensemble", y_test_raw, stack.predict(X_test_proc))


print("\n------- LEARNING CURVES --------")

def plot_learning_curve(model, name, filename):
    train_sizes, train_scores, val_scores=learning_curve(
        model, X_train_proc, y_train_log,
        cv=3, scoring='r2', n_jobs=-1,   # cv=3, 5 points only
        train_sizes=np.linspace(0.2, 1.0, 5)
    )
    train_mean=train_scores.mean(axis=1)
    train_std=train_scores.std(axis=1)
    val_mean=val_scores.mean(axis=1)
    val_std=val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training R2')
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.15, color='steelblue')
    plt.plot(train_sizes, val_mean, 'o-', color='darkorange', label='Validation R2')
    plt.fill_between(train_sizes,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.15, color='darkorange')
    plt.xlabel("Training Set Size")
    plt.ylabel("R2 Score (log scale)")
    plt.title(f"Learning Curve — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graphs_week8/{filename}", dpi=150)
    plt.close()
    print(f"Saved: {filename}")

    # Print convergence summary
    gap=train_mean[-1] - val_mean[-1]
    print(f"  Final train R2 : {train_mean[-1]:.4f}")
    print(f"  Final val   R2 : {val_mean[-1]:.4f}")
    print(f"  Train-val gap  : {gap:.4f} "
          f"({'possible overfit' if gap > 0.05 else 'good fit'})")

plot_learning_curve(best_rf,"RF Tuned","lc_rf_tuned.png")
plot_learning_curve(stack,"Stacking Ensemble","lc_stacking.png")


print("\n------- SHAP ANALYSIS -------")

er=shap.TreeExplainer(best_rf)
samp_id=np.random.RandomState(seed).choice(len(X_test_proc),size=150,replace=False)
samp_proc=X_test_proc[samp_id]
shap_vs=er.shap_values(samp_proc)

plt.figure(figsize=(10,6))
shap.summary_plot(shap_vs,samp_proc,feature_names=all_feature_names,plot_type="bar",show=False)
plt.title("SHAP Global Importance — RF Tuned (impact on log Yield)")
plt.tight_layout()
plt.savefig("graphs_week8/shap_global_bar.png",dpi=150)
plt.close()

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vs, samp_proc,feature_names=all_feature_names, show=False)
plt.title("SHAP Beeswarm — RF Tuned")
plt.tight_layout()
plt.savefig("graphs_week8/shap_summary_beeswarm.png", dpi=150)
plt.close()
print("Saved: shap_global_bar.png, shap_summary_beeswarm.png")

print("\n---------- CROP-WISE INSIGHTS --------")

insight_features=['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide',
                    'Fertilizer_per_Area', 'Pesticide_per_Area', 'Input_Intensity']

df['log_Yield']=np.log1p(df['Yield'])

crop_insight_rows=[]
for crop in sorted(df['Crop'].unique()):
    temp= df[df['Crop']== crop]
    if len(temp) < 30:
        continue
    corr=temp[insight_features + ['log_Yield']].corr()['log_Yield'].drop('log_Yield')
    top3= corr.abs().sort_values(ascending=False).head(3)
    print(f"\nCrop: {crop} (n={len(temp)})")
    for f in top3.index:
        direction="+" if corr[f] > 0 else "-"
        print(f" {f}: {corr[f]:.3f} ({direction})")
    crop_insight_rows.append({
        "Crop" : crop,
        "Top Feature": top3.index[0],
        "Correlation": round(corr[top3.index[0]], 3),
        "2nd Feature": top3.index[1] if len(top3) > 1 else "",
        "2nd Corr": round(corr[top3.index[1]], 3) if len(top3) > 1 else "",
        "Sample Size": len(temp)
    })

df.drop(columns=['log_Yield'],inplace=True)

crop_df=pd.DataFrame(crop_insight_rows)
crop_df.to_csv("graphs_week8/crop_insights.csv",index=False)
print(f"\nCrop insights saved: graphs_week8/crop_insights.csv")
print(f"Crops analysed:{len(crop_df)}")

top_feature_counts=crop_df['Top Feature'].value_counts()
plt.figure(figsize=(8, 4))
top_feature_counts.plot(kind='bar', color='mediumpurple', edgecolor='white')
plt.title("Most Common Top Driver Across All Crops")
plt.xlabel("Feature")
plt.ylabel("Number of Crops")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("graphs_week8/crop_top_drivers.png", dpi=150)
plt.close()
print("Saved: crop_top_drivers.png")


print("\n------- RESIDUAL ANALYSIS--------")

y_pred_log_test=stack.predict(X_test_proc)
y_pred_raw_test=np.expm1(y_pred_log_test)
res_log=y_test_log.values - y_pred_log_test
res_raw=y_test_raw.values - y_pred_raw_test

fig,axes=plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].axhline(0,color='red',linestyle='--')
axes[0,0].scatter(y_pred_log_test,res_log,alpha=0.3,s=10,color='steelblue')
axes[0,0].set_ylabel("Residual")
axes[0,0].set_xlabel("Predicted log(Yield)")
axes[0,0].set_title("Residuals vs Predicted — LOG scale (Stacking)")

axes[0,1].set_title("Residual Distribution — LOG scale")
axes[0,1].hist(res_log,bins=60,color='steelblue',edgecolor='white')
axes[0,1].set_xlabel("Residual (log scale)")
axes[0,1].set_title("Residual Distribution — LOG scale")

axes[1,0].scatter(y_pred_raw_test,res_raw,alpha=0.3,s=10,color='darkorange')
axes[1,0].set_title("Residuals vs Predicted — RAW scale (Stacking)")
axes[1,0].axhline(0,color='red',linestyle='--')
axes[1,0].set_xlabel("Predicted Yield (raw)")
axes[1,0].set_ylabel("Residual")


axes[1,1].hist(res_raw, bins=80, color='darkorange', edgecolor='white')
axes[1,1].set_title("Residual Distribution — RAW scale")
axes[1,1].set_xlabel("Residual (raw)")

plt.suptitle("Residual Analysis — Stacking Ensemble",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig("graphs_week8/residual_analysis.png",dpi=150)
plt.close()
print("Saved: residual_analysis.png")


all_results=pd.DataFrame([
    res_rf, res_gb, res_xgb,
    res_rf_tuned, res_xgb_tuned,
    res_stack
])

fig,axes=plt.subplots(1, 3, figsize=(16, 5))
for ax,metric,color in zip(axes,
                              ['R2_log', 'MAE', 'RMSE'],
                              ['steelblue', 'tomato', 'mediumseagreen']):
    bars=ax.bar(all_results['Model'], all_results[metric],
                  color=color, edgecolor='white')
    ax.set_title(f"{metric}")
    ax.set_xticklabels(all_results['Model'], rotation=30,
                       ha='right', fontsize=7)
    if metric=='R2_log':
        best_idx=all_results[metric].idxmax()
    else:
        best_idx=all_results[metric].idxmin()
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2.5)

plt.suptitle("Model Comparison — Week 8 Final", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("graphs_week8/model_comparison.png", dpi=150)
plt.close()
print("Saved: model_comparison.png")

final_pipeline=Pipeline([
    ('preprocessor', prepro),
    ('model', stack)
])
final_pipeline.fit(x_train, y_train_log)
sanity_r2=r2_score(y_test_raw,np.expm1(final_pipeline.predict(x_test)))
print(f"\nFinal pipeline R2 (raw scale): {sanity_r2:.4f}")
joblib.dump(final_pipeline, "final_model_week8.pkl")
print("Saved: final_model_week8.pkl")
best_r2_row=all_results.loc[all_results['R2_log'].idxmax()]
best_rmse_row=all_results.loc[all_results['RMSE'].idxmin()]

print("\n" + "="*55)
print("  WEEK 8 — FINAL PROJECT SUMMARY")
print("="*55)
print(f"\n{'Model':<30} {'R2 (log)':>9} {'R2 (raw)':>9} {'MAE':>9} {'RMSE':>9}")
print("-"*55)
for _, row in all_results.iterrows():
    marker=" BEST" if row['Model']== best_r2_row['Model'] else ""
    print(f"{row['Model']:<30} {row['R2_log']:>9.4f} {row['R2_raw']:>9.4f} "
        f"{row['MAE']:>9.2f} {row['RMSE']:>9.2f}{marker}")
print(f"""
Conclusion:
Best model by R2 (log scale):{best_r2_row['Model']}
  R2 log={best_r2_row['R2_log']:.4f}
  R2 raw={best_r2_row['R2_raw']:.4f}
  MAE={best_r2_row['MAE']:.2f}
  RMSE={best_r2_row['RMSE']:.2f}
Best model by RMS:{best_rmse_row['Model']}""")