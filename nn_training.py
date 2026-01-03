
## 神经网络训练
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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

# 确保数据标准化（因变量也需标准化）
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 定义神经网络模型
nn = MLPRegressor(
    max_iter=2000,
    random_state=42,
    early_stopping=True,  # 早停法
    validation_fraction=0.1,  # 10% 数据用于验证
    n_iter_no_change=50  # 早停耐心值
)

# 定义超参数网格
param_grid = {
    'hidden_layer_sizes': [(100, 50)],  # 简化网络结构
    'learning_rate_init': [0.01],  # 不同学习率
    'alpha': [0.01],  # L2 正则化
    'activation': ['relu']  # 激活函数
}

# 网格搜索
grid_search = GridSearchCV(nn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 最佳模型
best_nn = grid_search.best_estimator_
print("最佳超参数:", grid_search.best_params_)

# 交叉验证评估
cv_scores = cross_val_score(best_nn, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()  # 转换为正值

# 预测
y_pred_scaled = best_nn.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()  # 反标准化
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# 评估
test_mse = mean_squared_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

# 输出评估结果
print("\n评估结果:")
print(f"Cross-Validation MSE: {cv_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"R2: {r2:.4f}")