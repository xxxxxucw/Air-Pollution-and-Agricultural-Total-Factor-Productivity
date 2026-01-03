import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# 1. 加载数据
data = pd.read_csv("/Users/xuchengwei/Desktop/汇丰论文/代码处理/回归.ipynb/standardized_data.csv")
data["year"] = data["year"].astype(int)
data.set_index(["country_code", "year"], inplace=True)

# 2. 简化变量名
variable_mapping = {
    'pm2_5_concentration_ugm3': 'PM2.5',
    'surface_ozone_ppb': 'Ozone',
    'no2_concentration_ppb': 'NO2',
    'total_precipitation_mm': 'Precipitation',
    'wind_speed_10m_ms': 'Wind_Speed',
    'soil_temperature_level1_celsius': 'Soil_Temp',
    'surface_uvb_radiation_wm2': 'UVB_Radiation',
    'temperature_kelvin': 'Temperature',
    'total_cloud_cover_percent': 'Cloud_Cover',
    'gdp_per_capita_ppp': 'GDP_per_capita',
    'ppp_index': 'PPP_Index',
    'foreign_direct_investment_gdp_ratio_percent': 'FDI_Ratio',
    'tourism_revenue_gdp_ratio_percent': 'Tourism_Ratio',
    'population_total': 'Population',
    'rural_employment_rate_percent': 'Rural_Employment',
    'rural_infrastructure_index': 'Rural_Infrastructure',
    'environmental_regulation_quality_index': 'Env_Regulation',
    'total_co2_emissions_kt': 'CO2_Emissions'
}
data = data.rename(columns=variable_mapping)
core_vars = ['PM2.5', 'Ozone', 'NO2']
control_vars = [
    'Precipitation', 'Wind_Speed', 'Soil_Temp', 'UVB_Radiation', 'Temperature', 'Cloud_Cover',
    'GDP_per_capita', 'PPP_Index', 'FDI_Ratio', 'Tourism_Ratio', 'Population',
    'Rural_Employment', 'Rural_Infrastructure', 'Env_Regulation', 'CO2_Emissions'
]
X = data[core_vars + control_vars]
y = data['total_factor_productivity']

# 3. 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# 4. 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_test_df = pd.DataFrame(X_test, columns=X.columns)  # 转换为 DataFrame 以便后续处理

# 5. 剔除极端点（基于特征值的 1% 和 99% 分位数）
def remove_outliers(df, columns, lower_quantile=0.01, upper_quantile=0.99):
    mask = pd.Series(True, index=df.index)
    for col in columns:
        lower_bound = df[col].quantile(lower_quantile)
        upper_bound = df[col].quantile(upper_quantile)
        mask = mask & (df[col] >= lower_bound) & (df[col] <= upper_bound)
    return mask

outlier_mask = remove_outliers(X_test_df, core_vars)
X_test_clean = X_test[outlier_mask]
X_test_clean_df = X_test_df[outlier_mask]
y_test_clean = y_test[outlier_mask]
print(f"原始测试集样本数: {len(X_test)}")
print(f"剔除极端点后样本数: {len(X_test_clean)}")
print(f"剔除比例: {(1 - len(X_test_clean) / len(X_test)) * 100:.2f}%")

# 6. 训练随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 7. 模型性能评估
y_pred_scaled = rf.predict(X_test_clean)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_orig = scaler_y.inverse_transform(y_test_clean.reshape(-1, 1)).ravel()
mse = mean_squared_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)
print("Random Forest 性能（剔除极端点后）:")
print(f"Test MSE: {mse:.4f}")
print(f"R2: {r2:.4f}")
print("Feature Importance (Core Variables):")
print(pd.Series(rf.feature_importances_, index=X.columns)[core_vars])

# 8. SHAP 值计算
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test_clean)

# 9. SHAP 全局重要性图
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_clean, feature_names=X.columns, show=False, plot_size=None)
plt.title("SHAP Feature Importance for Random Forest", fontsize=14)
plt.xlabel("Mean |SHAP Value|", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('shap_summary_rf_clean.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. SHAP 局部依赖图（PM2.5）
plt.figure(figsize=(8, 6))
shap.dependence_plot('PM2.5', shap_values, X_test_clean, feature_names=X.columns, show=False, 
                     dot_size=10, alpha=0.6, cmap=plt.cm.viridis)
plt.title("SHAP Dependence Plot for PM2.5", fontsize=14)
plt.xlabel("PM2.5 (Standardized)", fontsize=12)
plt.ylabel("SHAP Value for PM2.5", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('shap_dependence_pm25_rf_clean.png', dpi=300, bbox_inches='tight')
plt.close()

# 11. SHAP 局部依赖图（Ozone）
plt.figure(figsize=(8, 6))
shap.dependence_plot('Ozone', shap_values, X_test_clean, feature_names=X.columns, show=False, 
                     dot_size=10, alpha=0.6, cmap=plt.cm.viridis)
plt.title("SHAP Dependence Plot for Ozone", fontsize=14)
plt.xlabel("Ozone (Standardized)", fontsize=12)
plt.ylabel("SHAP Value for Ozone", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('shap_dependence_ozone_rf_clean.png', dpi=300, bbox_inches='tight')
plt.close()

# 12. SHAP 局部依赖图（NO2）
plt.figure(figsize=(8, 6))
shap.dependence_plot('NO2', shap_values, X_test_clean, feature_names=X.columns, show=False, 
                     dot_size=10, alpha=0.6, cmap=plt.cm.viridis)
plt.title("SHAP Dependence Plot for NO2", fontsize=14)
plt.xlabel("NO2 (Standardized)", fontsize=12)
plt.ylabel("SHAP Value for NO2", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('shap_dependence_no2_rf_clean.png', dpi=300, bbox_inches='tight')
plt.close()

# 13. SHAP 交互效应图（PM2.5 与 Temperature）
plt.figure(figsize=(8, 6))
shap.dependence_plot('PM2.5', shap_values, X_test_clean, feature_names=X.columns, 
                     interaction_index='Temperature', show=False, dot_size=10, alpha=0.6, cmap=plt.cm.viridis)
plt.title("SHAP Dependence Plot for PM2.5 with Temperature Interaction", fontsize=14)
plt.xlabel("PM2.5 (Standardized)", fontsize=12)
plt.ylabel("SHAP Value for PM2.5", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('shap_dependence_pm25_temperature_rf_clean.png', dpi=300, bbox_inches='tight')
plt.close()

# 14. 个体样本分析（选择第一个非极端样本）
sample_idx = 0
plt.figure(figsize=(12, 4))
shap.force_plot(explainer.expected_value, shap_values[sample_idx], X_test_clean[sample_idx], 
                feature_names=X.columns, matplotlib=True, show=False, text_rotation=45)

plt.tight_layout()
plt.savefig('shap_force_plot_rf_clean.png', dpi=300, bbox_inches='tight')
plt.close()

# 14. 个体样本分析（选择第一个非极端样本）
sample_idx = 0

plt.figure(figsize=(12, 4))
shap_plot = shap.force_plot(explainer.expected_value, 
                           shap_values[sample_idx], 
                           X_test_clean[sample_idx], 
                           feature_names=X.columns, 
                           matplotlib=True, 
                           show=False, 
                           text_rotation=45)

# 获取当前的Axes对象
ax = plt.gca()

# 遍历所有的文本对象，查找并修改"变量=值"格式的标签
for text in ax.texts:
    txt = text.get_text()
    if '=' in txt:
        parts = txt.split('=')
        if len(parts) == 2:
            try:
                # 尝试将等号后部分转为浮点数并保留两位小数
                value = float(parts[1])
                new_value = f"{value:.2f}"
                new_text = f"{parts[0]}={new_value}"
                text.set_text(new_text)
            except ValueError:
                # 如果转换失败（非数字），则保持原样
                pass

plt.title(f"SHAP Force Plot for Sample {sample_idx}", fontsize=14)
plt.tight_layout()
plt.savefig('shap_force_plot_rf_clean.png', dpi=300, bbox_inches='tight')
plt.close()