import pandas as pd
from linearmodels.panel import PanelOLS
import numpy as np

# ======================
# 1. 数据准备与预处理
# ======================
# 假设数据已加载为data，包含以下列：
# - 索引：country_code（国家代码）、year（年份）
# - 因变量：total_factor_productivity（TFPit）
# - 核心自变量：pm2_5_concentration_ugm3（PM2.5）、surface_ozone_ppb（O3）、no2_concentration_ppb（NO2）
# - 控制变量：Xit（如代码中control_vars定义）
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 读取Excel数据（需确保路径正确，若提示权限问题可尝试添加engine='openpyxl'）
data = pd.read_csv("/Users/xuchengwei/Desktop/汇丰论文/代码处理/回归.ipynb/standardized_data.csv")

# 转换年份为整数（原数据year为浮点型，如1990.0→1990）
data["year"] = data["year"].astype(int)

# 设置面板数据索引（国家代码+年份）
data.set_index(["country_code", "year"], inplace=True)


# 定义变量
dependent_var = 'total_factor_productivity'  # TFPit
core_vars = ['pm2_5_concentration_ugm3', 'surface_ozone_ppb', 'no2_concentration_ppb','total_co2_emissions_kt']  # 核心自变量
control_vars = [
    'gdp_per_capita_ppp', 'temperature', 'total_precipitation_mm','surface_uvb_radiation_wm2',
    'rural_employment_rate_percent', 'rural_infrastructure_index',
    'environmental_regulation_quality_index','population_total','surface_air_pressure_hpa','tourism_revenue_gdp_ratio_percent'
]  # 控制变量向量Xit

# 合并所有自变量（核心+控制）
all_exog_vars = core_vars + control_vars


# ======================
# 2. 基准回归模型（含双向固定效应）
# ======================
def run_base_model(data):
    """运行基准回归模型"""
    formula = f"{dependent_var} ~ 1+ {' + '.join(all_exog_vars)} + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=data, drop_absorbed=True)
    result = model.fit()
    return result

# 运行基准模型
print("=== 基准模型（双向固定效应） ===")
result_base = run_base_model(data)
print(result_base.summary)


# ======================
# 3. 分组回归检验H1a（按TFP高低分组）
# ======================
def group_regression_by_tfp(data, threshold='median'):
    """按TFP水平分组回归"""
    if threshold == 'median':
        tfp_threshold = data[dependent_var].median()
    elif threshold == 'mean':
        tfp_threshold = data[dependent_var].mean()
    else:
        raise ValueError("threshold需为'median'或'mean'")
    
    # 划分高低TFP组
    data_low = data[data[dependent_var] <= tfp_threshold]
    data_high = data[data[dependent_var] > tfp_threshold]
    
    # 运行分组模型
    print(f"\n=== 低TFP组（<= {tfp_threshold:.2f}） ===")
    result_low = run_base_model(data_low)
    print(result_low.summary)
    
    print(f"\n=== 高TFP组（> {tfp_threshold:.2f}） ===")
    result_high = run_base_model(data_high)
    print(result_high.summary)
    
    return result_low, result_high

# 按中位数分组检验H1a
result_low, result_high = group_regression_by_tfp(data, threshold='median')


# ======================
# 4. 交互项模型检验H1b（TFP×污染）
# ======================
def interact_model_with_tfp(data):
    """引入TFP与核心自变量的交互项"""
    # 创建交互项（TFP × 污染物）
    for var in core_vars:
        data[f"{var}_tfp_interact"] = data[var] * data[dependent_var]
    
    # 构建含交互项的公式
    interact_vars = [f"{var}_tfp_interact" for var in core_vars]
    formula = f"""
    {dependent_var} ~ 1 + 
    {' + '.join(all_exog_vars)} + 
    {' + '.join(interact_vars)} + 
    EntityEffects + TimeEffects
    """
    
    # 拟合模型
    print("\n=== 交互项模型（检验H1b） ===")
    model_interact = PanelOLS.from_formula(formula, data=data, drop_absorbed=True)
    result_interact = model_interact.fit()
    print(result_interact.summary)
    return result_interact

# 运行交互项模型
result_interact = interact_model_with_tfp(data)