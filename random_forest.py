# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 读取CSV文件
data = pd.read_csv('bam_features.csv')

# 特征列和目标列
X = data[['mean_coverage', 'stddev_coverage', 'peak_coverage', 'repeat_fraction']]
y = data[['window_size', 'threshold', 'step_size', 'rd']]

# 初始化随机森林回归模型
rf_window_size = RandomForestRegressor(n_estimators=100, random_state=42)
rf_threshold = RandomForestRegressor(n_estimators=100, random_state=42)
rf_step_size = RandomForestRegressor(n_estimators=100, random_state=42)
rf_rd = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_window_size.fit(X, y['window_size'])
rf_threshold.fit(X, y['threshold'])
rf_step_size.fit(X, y['step_size'])
rf_rd.fit(X, y['rd'])

# 模型训练完成后可以保存模型
import joblib
joblib.dump(rf_window_size, 'rf_window_size_model.pkl')
joblib.dump(rf_threshold, 'rf_threshold_model.pkl')
joblib.dump(rf_step_size, 'rf_step_size_model.pkl')
joblib.dump(rf_rd, 'rf_rd_model.pkl')

print("successful")
