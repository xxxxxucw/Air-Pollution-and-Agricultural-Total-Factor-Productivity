import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei",  "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class RegionalHeterogeneityAnalyzer:
    def __init__(self, file_path):
        """初始化分析器"""
        self.file_path = file_path
        self.data = None
        self.clustered_data = None
        
    def load_data(self):
        """加载数据"""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"数据加载成功，共{self.data.shape[0]}行，{self.data.shape[1]}列")
            return True
        except FileNotFoundError:
            print(f"错误：文件未找到 - {self.file_path}")
            return False
        except Exception as e:
            print(f"错误：加载数据时出错 - {str(e)}")
            return False
    
    def preprocess_data(self):
        """数据预处理"""
        # 移除包含缺失值的行
        self.data = self.data.dropna()
        print(f"处理后的数据共{self.data.shape[0]}行，{self.data.shape[1]}列")
        
        # 基于分位数创建收入水平分类
        income_quantile = self.data['gdp_per_capita_ppp'].quantile(0.8)
        self.data['income_level'] = np.where(
            self.data['gdp_per_capita_ppp'] >= income_quantile, 
            '高收入', '低收入'
        )
        
        # 基于分位数创建农业TFP水平分类
        atfp_quantiles = self.data['total_factor_productivity'].quantile([1/4, 3/4])
        self.data['atfp_level'] = pd.cut(
            self.data['total_factor_productivity'],
            bins=[-np.inf, atfp_quantiles[1/4], atfp_quantiles[3/4], np.inf],
            labels=['低', '中', '高']
        )
        
        # 基于温度创建气候带分类 (假设温度阈值为298K (25°C))
        self.data['climate_zone'] = np.where(
            self.data['temperature_kelvin'] >= self.data["temperature_kelvin"].quantile(0.9), 
            '热带', '温带'
        )
        
        print("数据预处理完成")
    
    def perform_two_step_clustering(self):
        """执行两步聚类分析"""
        # 提取用于聚类的特征
        cluster_features = [
            'gdp_per_capita_ppp', 'total_factor_productivity', 
            'temperature_kelvin', 'environmental_regulation_quality_index'
        ]
        
        # 标准化特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.data[cluster_features])
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=6, random_state=42)
        self.data['cluster'] = kmeans.fit_predict(scaled_features)
        
        # 为聚类分配有意义的标签
        cluster_labels = {
            0: "低TFP-低收入-热带",
            1: "中TFP-中等收入-温带",
            2: "高TFP-高收入-温带",
            3: "中TFP-中等收入-热带",
            4: "低TFP-低收入-温带",
            5: "高TFP-高收入-热带"
        }
        
        self.data['cluster_label'] = self.data['cluster'].map(cluster_labels)
        self.clustered_data = self.data.copy()
        
        print("两步聚类分析完成")
        print("聚类分布:")
        print(self.data['cluster_label'].value_counts())
        
        return self.clustered_data
    
    def analyze_group_differences(self):
        """分析不同聚类组间的差异"""
        if self.clustered_data is None:
            print("错误：请先执行聚类分析")
            return None
        
        # 准备结果数据框
        results = pd.DataFrame(columns=['污染物', '聚类组1', '聚类组2', '均值差异', 'p值', '显著性'])
        
        # 定义要比较的污染物
        pollutants = ['pm2_5_concentration_ugm3', 'surface_ozone_ppb', 'no2_concentration_ppb']
        
        # 获取所有聚类组
        cluster_groups = self.clustered_data['cluster_label'].unique()
        
        # 对每个污染物进行组间差异检验
        comparison_index = 0
        for pollutant in pollutants:
            for i in range(len(cluster_groups)):
                for j in range(i+1, len(cluster_groups)):
                    group1 = self.clustered_data[self.clustered_data['cluster_label'] == cluster_groups[i]]
                    group2 = self.clustered_data[self.clustered_data['cluster_label'] == cluster_groups[j]]
                    
                    # 执行t检验
                    t_stat, p_value = stats.ttest_ind(
                        group1[pollutant], 
                        group2[pollutant],
                        equal_var=False  # 假设方差不相等
                    )
                    
                    # 记录结果
                    results.loc[comparison_index] = [
                        pollutant,
                        cluster_groups[i],
                        cluster_groups[j],
                        group1[pollutant].mean() - group2[pollutant].mean(),
                        p_value,
                        "*" if p_value < 0.05 else ""
                    ]
                    comparison_index += 1
        
        # 执行ANOVA检验 (检验所有组之间的差异)
        for pollutant in pollutants:
            # 准备ANOVA数据
            anova_data = pd.DataFrame()
            for group in cluster_groups:
                group_data = self.clustered_data[self.clustered_data['cluster_label'] == group]
                anova_data[group] = group_data[pollutant].reset_index(drop=True)
            
            # 重塑数据以适应statsmodels格式
            anova_data_long = pd.melt(anova_data.reset_index(), id_vars=['index'], value_vars=cluster_groups)
            anova_data_long.columns = ['index', 'cluster', 'value']
            
            # 执行ANOVA
            model = ols('value ~ C(cluster)', data=anova_data_long).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # 添加ANOVA结果到结果数据框
            results.loc[comparison_index] = [
                f"{pollutant} (ANOVA)",
                "所有组",
                "",
                "",
                anova_table['PR(>F)'][0],
                "*" if anova_table['PR(>F)'][0] < 0.05 else ""
            ]
            comparison_index += 1
        
        return results
    
    def visualize_results(self):
        """可视化分析结果"""
        if self.clustered_data is None:
            print("错误：请先执行聚类分析")
            return
        
        # 创建分析结果保存目录
        if not os.path.exists('analysis_results'):
            os.makedirs('analysis_results')
        
        # 1. 绘制聚类分布
        plt.figure(figsize=(12, 6))
        sns.countplot(x='cluster_label', data=self.clustered_data)
        plt.title('聚类组分布')
        plt.xlabel('聚类组')
        plt.ylabel('国家数量')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('analysis_results/cluster_distribution.png')
        plt.close()
        
        # 2. 绘制不同聚类组的污染物浓度比较
        pollutants = ['pm2_5_concentration_ugm3', 'surface_ozone_ppb', 'no2_concentration_ppb']
        pollutant_names = {'pm2_5_concentration_ugm3': 'PM2.5浓度 (μg/m³)', 
                           'surface_ozone_ppb': '臭氧浓度 (ppb)',
                           'no2_concentration_ppb': '二氧化氮浓度 (ppb)'}
        
        for pollutant in pollutants:
            plt.figure(figsize=(14, 6))
            sns.boxplot(x='cluster_label', y=pollutant, data=self.clustered_data)
            plt.title(f'不同聚类组的{pollutant_names[pollutant]}比较')
            plt.xlabel('聚类组')
            plt.ylabel(pollutant_names[pollutant])
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'analysis_results/{pollutant}_comparison.png')
            plt.close()
        
        # 3. 绘制热图展示聚类特征关系
        cluster_features = [
            'gdp_per_capita_ppp', 'total_factor_productivity', 'temperature_kelvin', 
            'environmental_regulation_quality_index', 'pm2_5_concentration_ugm3',
            'surface_ozone_ppb', 'no2_concentration_ppb'
        ]
        
        plt.figure(figsize=(12, 10))
        cluster_mean = self.clustered_data.groupby('cluster_label')[cluster_features].mean()
        sns.heatmap(cluster_mean, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('聚类组特征均值热图')
        plt.tight_layout()
        plt.savefig('analysis_results/cluster_features_heatmap.png')
        plt.close()
        
        print("可视化结果已保存至 'analysis_results' 目录")
    
    def test_hypotheses(self):
        """测试研究假设"""
        if self.clustered_data is None:
            print("错误：请先执行聚类分析")
            return None
        
        # H2a：热带地区臭氧对 TFP 的抑制效应比温带地区更显著
        tropical_data = self.clustered_data[self.clustered_data['climate_zone'] == '热带']
        temperate_data = self.clustered_data[self.clustered_data['climate_zone'] == '温带']
        
        # 热带地区臭氧对TFP的回归
        X_tropical = sm.add_constant(tropical_data[['surface_ozone_ppb', 'gdp_per_capita_ppp']])
        y_tropical = tropical_data['total_factor_productivity']
        model_tropical = sm.OLS(y_tropical, X_tropical).fit()
        
        # 温带地区臭氧对TFP的回归
        X_temperate = sm.add_constant(temperate_data[['surface_ozone_ppb', 'gdp_per_capita_ppp']])
        y_temperate = temperate_data['total_factor_productivity']
        model_temperate = sm.OLS(y_temperate, X_temperate).fit()
        
        # H2b：低收入国家 PM2.5 对 TFP 的影响强度高于高收入国家，且该差异与污染治理投入呈负相关
        low_income_data = self.clustered_data[self.clustered_data['income_level'] == '低收入']
        high_income_data = self.clustered_data[self.clustered_data['income_level'] == '高收入']
        
        # 低收入国家PM2.5对TFP的回归
        X_low_income = sm.add_constant(low_income_data[['pm2_5_concentration_ugm3', 'gdp_per_capita_ppp']])
        y_low_income = low_income_data['total_factor_productivity']
        model_low_income = sm.OLS(y_low_income, X_low_income).fit()
        
        # 高收入国家PM2.5对TFP的回归
        X_high_income = sm.add_constant(high_income_data[['pm2_5_concentration_ugm3', 'gdp_per_capita_ppp']])
        y_high_income = high_income_data['total_factor_productivity']
        model_high_income = sm.OLS(y_high_income, X_high_income).fit()
        
        # 检验污染治理投入与影响差异的关系
        # 创建一个新变量表示PM2.5影响差异
        self.clustered_data['pm25_effect'] = self.clustered_data['pm2_5_concentration_ugm3'] * self.clustered_data['total_factor_productivity']
        
        # 分析污染治理投入与PM2.5影响的关系
        X_regulation = sm.add_constant(self.clustered_data[['environmental_regulation_quality_index', 'gdp_per_capita_ppp']])
        y_pm25_effect = self.clustered_data['pm25_effect']
        model_regulation = sm.OLS(y_pm25_effect, X_regulation).fit()
        
        # 整理假设检验结果
        hypotheses_results = {
            "H2a": {
                "热带地区臭氧系数": model_tropical.params['surface_ozone_ppb'],
                "热带地区臭氧p值": model_tropical.pvalues['surface_ozone_ppb'],
                "温带地区臭氧系数": model_temperate.params['surface_ozone_ppb'],
                "温带地区臭氧p值": model_temperate.pvalues['surface_ozone_ppb'],
                "结论": "支持" if (model_tropical.params['surface_ozone_ppb'] < model_temperate.params['surface_ozone_ppb'] 
                              and model_tropical.pvalues['surface_ozone_ppb'] < 0.05) else "不支持"
            },
            "H2b": {
                "低收入国家PM2.5系数": model_low_income.params['pm2_5_concentration_ugm3'],
                "低收入国家PM2.5p值": model_low_income.pvalues['pm2_5_concentration_ugm3'],
                "高收入国家PM2.5系数": model_high_income.params['pm2_5_concentration_ugm3'],
                "高收入国家PM2.5p值": model_high_income.pvalues['pm2_5_concentration_ugm3'],
                "污染治理投入系数": model_regulation.params['environmental_regulation_quality_index'],
                "污染治理投入p值": model_regulation.pvalues['environmental_regulation_quality_index'],
                "结论": "支持" if (model_low_income.params['pm2_5_concentration_ugm3'] < model_high_income.params['pm2_5_concentration_ugm3'] 
                              and model_low_income.pvalues['pm2_5_concentration_ugm3'] < 0.05
                              and model_regulation.params['environmental_regulation_quality_index'] > 0
                              and model_regulation.pvalues['environmental_regulation_quality_index'] < 0.05) else "不支持"
            }
        }
        
        return hypotheses_results

# 主函数
def main():
    # 文件路径
    file_path = '/Users/xuchengwei/Desktop/汇丰论文/代码处理/回归.ipynb/standardized_data.csv'
    
    # 创建分析器实例
    analyzer = RegionalHeterogeneityAnalyzer(file_path)
    
    # 加载数据
    if analyzer.load_data():
        os.makedirs('analysis_results', exist_ok=True)  # exist_ok=True 确保目录存在时不报错
        # 数据预处理
        analyzer.preprocess_data()
        
        # 执行两步聚类
        clustered_data = analyzer.perform_two_step_clustering()
        
        # 分析组间差异
        group_differences = analyzer.analyze_group_differences()
        if group_differences is not None:
            print("\n组间差异检验结果:")
            print(group_differences)
            group_differences.to_csv('analysis_results/group_differences.csv', index=False)
        
        # 可视化结果
        analyzer.visualize_results()
        
        # 测试研究假设
        hypotheses_results = analyzer.test_hypotheses()
        if hypotheses_results is not None:
            print("\n假设检验结果:")
            for hypothesis, result in hypotheses_results.items():
                print(f"\n{hypothesis}:")
                for key, value in result.items():
                    print(f"  {key}: {value}")
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()    