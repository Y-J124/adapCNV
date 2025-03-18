# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ��ȡCSV�ļ�
data = pd.read_csv('bam_features.csv')

# �����к�Ŀ����
X = data[['mean_coverage', 'stddev_coverage', 'peak_coverage', 'repeat_fraction']]
y = data[['window_size', 'threshold', 'step_size', 'rd']]

# ��ʼ�����ɭ�ֻع�ģ��
rf_window_size = RandomForestRegressor(n_estimators=100, random_state=42)
rf_threshold = RandomForestRegressor(n_estimators=100, random_state=42)
rf_step_size = RandomForestRegressor(n_estimators=100, random_state=42)
rf_rd = RandomForestRegressor(n_estimators=100, random_state=42)

# ѵ��ģ��
rf_window_size.fit(X, y['window_size'])
rf_threshold.fit(X, y['threshold'])
rf_step_size.fit(X, y['step_size'])
rf_rd.fit(X, y['rd'])

# ģ��ѵ����ɺ���Ա���ģ��
import joblib
joblib.dump(rf_window_size, 'rf_window_size_model.pkl')
joblib.dump(rf_threshold, 'rf_threshold_model.pkl')
joblib.dump(rf_step_size, 'rf_step_size_model.pkl')
joblib.dump(rf_rd, 'rf_rd_model.pkl')

print("successful")
