import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import uuid

# 加载数据
data = pd.read_csv("/Users/xuchengwei/Desktop/汇丰论文/代码处理/回归.ipynb/standardized_data.csv")
data["year"] = data["year"].astype(int)
data.set_index(["country_code", "year"], inplace=True)

# 定义因变量和自变量
y = data['total_factor_productivity']
core_vars = ['pm2_5_concentration_ugm3', 'surface_ozone_ppb', 'no2_concentration_ppb']
control_vars = [
    'total_precipitation_mm', 'wind_speed_10m_ms', 'soil_temperature_level1_celsius',
    'surface_uvb_radiation_wm2', 'temperature_kelvin', 'total_cloud_cover_percent',
    'gdp_per_capita_ppp', 'ppp_index', 'foreign_direct_investment_gdp_ratio_percent',
    'tourism_revenue_gdp_ratio_percent', 'population_total', 
    'rural_employment_rate_percent', 'rural_infrastructure_index',
    'environmental_regulation_quality_index', 'total_co2_emissions_kt'
]
X = data[core_vars + control_vars]

# 标准化数据
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 初始化模型和结果存储
models = {
    'LASSO': LassoCV(cv=5, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'CatBoost': CatBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=0),
    'NeuralNetwork': MLPRegressor(
        hidden_layer_sizes=(50,), max_iter=2000, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50
    )
}
results = {}

# 训练和评估模型
for name, model in models.items():
    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    
    # 训练模型
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    # 评估
    mse = mean_squared_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    
    # 存储结果
    results[name] = {
        'CV_MSE': cv_mse,
        'Test_MSE': mse,
        'R2': r2,
        'Feature Importance': None
    }
    
    # 获取特征重要性（仅适用于支持的模型）
    if name in ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost']:
        results[name]['Feature Importance'] = pd.Series(
            model.feature_importances_, index=X.columns
        )
    elif name == 'LASSO':
        results[name]['Feature Importance'] = pd.Series(model.coef_, index=X.columns)

# 打印性能比较
print("模型性能比较：")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"Cross-Validation MSE: {metrics['CV_MSE']:.4f}")
    print(f"Test MSE: {metrics['Test_MSE']:.4f}")
    print(f"R2: {metrics['R2']:.4f}")
    if metrics['Feature Importance'] is not None:
        print("Feature Importance (Core Variables):")
        print(metrics['Feature Importance'][core_vars])

