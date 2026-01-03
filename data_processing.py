import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 1. 读取数据
file_path = 'C:/Users/Administrator/Desktop/数据分析作业/1_data/final_data.csv'
try:
    df = pd.read_csv(file_path)
    print("数据已成功加载，形状：", df.shape)
except Exception as e:
    print(f"数据加载失败: {e}")
    exit(1)

# 2. 定义列的处理方式

log_transform_columns = [
    'total_factor_productivity',
    'gdp_per_capita_ppp',
    'gdp_current_usd',
    'pm2_5_concentration_ugm3',
    'surface_ozone_ppb',
    'no2_concentration_ppb',  
    'total_precipitation_mm',
    'agricultural_co2_emissions_kt',
    'total_co2_emissions_kt',
    'population_total',
    'per_capita_health_expenditure_usd'
]


z_score_columns = [
    'temperature_2m_celsius',
    'leaf_area_index_high_vegetation',
    'snow_depth_cm',
    'snowfall_mm',
    'wind_speed_10m_ms',
    'surface_air_pressure_hpa',
    'surface_runoff_mm',
    'soil_temperature_level1_celsius',
    'total_cloud_cover_percent',
    'surface_uvb_radiation_wm2',
    'ppp_index',
    'environmental_regulation_quality_index',
    'tourism_revenue_gdp_ratio_percent',
    'rural_employment_rate_percent',
    'rural_infrastructure_index',
    'rural_population_percent',
    'rural_school_enrollment_rate_percent',
    'rural_electrification_rate_percent',
    'unemployment_rate_percent',
    'inflation_rate_year_on_year_percent',
    'foreign_direct_investment_gdp_ratio_percent'
]

# 3. 执行对数转换 (Log Transformation)
print("\n正在进行对数转换...")
for col in log_transform_columns:
    if col in df.columns:
        # 检查是否有零值或负值
        min_val = df[col].min()
        if min_val <= 0:
            # Shifted Log: 确保数据整体平移到 > 0，且 +1 保证 log(1)=0
            # 记录 shift 值以便知晓
            shift_val = abs(min_val) + 1
            df[col] = np.log(df[col] + shift_val)
            print(f"列 {col} (含<=0值) -> 平移 {shift_val:.2f} 后取对数")
        else:
            df[col] = np.log(df[col])
            print(f"列 {col} -> 直接取对数")
    else:
        print(f"警告：列 {col} 不存在，跳过")

# 4. 执行 Z-score 标准化 (含异常值处理)
print("\n正在进行Z-score标准化...")
scaler = StandardScaler()

for col in z_score_columns:
    if col in df.columns:
        # 4.1 异常值检测与截断 (Winsorization)

        z_scores_temp = np.abs(stats.zscore(df[col]))
        outlier_count = (z_scores_temp > 3).sum()
        
        if outlier_count > 0:
            lower_limit = df[col].quantile(0.01)
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
            print(f"列 {col} -> 检测到 {outlier_count} 个异常值，已截断至 [1%, 99%] 分位数")
        
        # 4.2 标准化
        df[col] = scaler.fit_transform(df[[col]])
        # print(f"列 {col} -> Z-score 完成") # 减少刷屏，可选注释
    else:
        print(f"警告：列 {col} 不存在，跳过")

# 5. 检查是否有未处理的数值列 (防止漏网之鱼)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
processed_cols = set(log_transform_columns + z_score_columns + ['year']) # 假设year不处理
remaining = [c for c in numeric_cols if c not in processed_cols]
if remaining:
    print(f"\n：{remaining}")

# 6. 保存
output_path = "transformed_data_final.csv"
df.to_csv(output_path, index=False)
print(f"\n处理完成！数据已保存至：{output_path}")
