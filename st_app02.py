import streamlit as st  
import seaborn as sns  
from sklearn.linear_model import LinearRegression  

# tips 데이터셋 불러오기  
tips = sns.load_dataset("tips")  

# Streamlit 앱 설정  
st.title("Tip Prediction App")  

# 입력 데이터 받기  
total_bill = st.number_input("Enter the total bill amount:", min_value=0.0, step=0.1)  

# 선형 회귀 모델 생성 및 학습  
X = tips['total_bill'].values.reshape(-1, 1)  
y = tips['tip'].values  
model = LinearRegression()  
model.fit(X, y)  

# 예측 결과 출력  
prediction = model.predict([[total_bill]])  
st.write(f"The predicted tip amount is: ${prediction[0]:.2f}")  
