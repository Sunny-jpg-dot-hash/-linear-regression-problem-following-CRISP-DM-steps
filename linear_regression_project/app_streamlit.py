import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 生成資料的函數
def generate_data(a=2, b=3, noise=1, n_points=20):
    np.random.seed(42)
    X = np.random.rand(n_points) * 10
    y = a * X + b + np.random.randn(n_points) * noise
    return pd.DataFrame({'X': X, 'y': y})

# 訓練模型的函數
def train_model(data):
    X = data[['X']]
    y = data['y']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Streamlit 介面
st.title("線性回歸模型互動應用")
st.write("使用滑桿來調整模型參數，並觀察線性回歸的結果。")

# 用戶輸入參數
a = st.slider('斜率 (a)', 0.0, 10.0, 2.0)
b = st.slider('截距 (b)', 0.0, 10.0, 3.0)
noise = st.slider('雜訊強度', 0.0, 5.0, 1.0)
n_points = st.slider('資料點數', 10, 100, 20)

# 生成資料
data = generate_data(a, b, noise, n_points)
st.write("### 生成的資料", data)

# 訓練模型並顯示結果
model = train_model(data)
st.write(f"**模型斜率 (a):** {model.coef_[0]}")
st.write(f"**模型截距 (b):** {model.intercept_}")

# 繪製回歸結果
fig, ax = plt.subplots()
ax.scatter(data['X'], data['y'], color='blue', label='資料點')
ax.plot(data['X'], model.predict(data[['X']]), color='red', label='回歸線')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()
st.pyplot(fig)
