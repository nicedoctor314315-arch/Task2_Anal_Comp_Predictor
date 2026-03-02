import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('Task2_Anal_Comp_pred.pkl')  # 加载训练好的RF模型

# Streamlit UI
st.title("Anal Complications Predictor for CD patient in coming 12 months")

# Sidebar for input options
st.sidebar.header("Input Sample Data")  # 侧边栏输入样本数据

# Course of diagnosis(确诊病程)
cod = st.sidebar.number_input("Course of diagnosis (Months):", min_value=0, max_value=600, value=0)

# Duration input
dur = st.sidebar.number_input("Duration:", min_value=0, max_value=600, value=0)

# Age of onset (确诊年龄)
AOO = st.sidebar.number_input("Age of onset:", min_value=1, max_value=120, value=50)

# Smoking History
Smoke = st.sidebar.selectbox("Smoking History:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Alcohol Consumption History
ACH = st.sidebar.selectbox("Alcohol Consumption History:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Gastrointestinal_Surgery_History input
GSH = st.sidebar.selectbox("Gastrointestinal Surgery History:",  options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Diarrhea input
Dia = st.sidebar.selectbox("Diarrhea:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Gastrointestinal Bleeding input
GB = st.sidebar.selectbox("Gastrointestinal Bleeding:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Vomitting input
Vom = st.sidebar.selectbox("Vomitting:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Abdominal_Complication_History input
ABCH = st.sidebar.selectbox("Abdominal Complication History:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Anal_Complication_History input
ANCH = st.sidebar.selectbox("Anal Complication History:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Frequency of Loose Stools
FOLS = st.sidebar.number_input("Frequency of Loose Stools\n(Consecutive 7 days):", min_value=0, max_value=600, value=0)

# Extra-intestinal Manifestations
EIM = st.sidebar.selectbox("Extra-intestinal Manifestations\n(肠外表现并发症):", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# CDAI Score
CDAI = st.sidebar.number_input("CDAI Score:", min_value=0, max_value=1000, value=0)

# Activity
ACT = st.sidebar.selectbox("Activity:", options=[0, 1], format_func=lambda x: 'Active Disease (1)' if x == 0 else 'Remission (0)')

# RBC,HB,MCV,CRP(mg/l),TP,Urea Nitrogen,Serum Potassium,Uric Acid,Folic Acid (nmol/L),vitB12
RBC = st.sidebar.number_input("RBC (*10^12):", min_value=0, max_value=100, value=0)
HB = st.sidebar.number_input("HB (g/L):", min_value=0, max_value=1000, value=0)
MCV = st.sidebar.number_input("MCV (fL):", min_value=0, max_value=1000, value=0)
CRP = st.sidebar.number_input("CRP (mg/L):", min_value=0, max_value=1000, value=0)
TP = st.sidebar.number_input("Total Protein (g/L):", min_value=0, max_value=1000, value=0)
UN = st.sidebar.number_input("Urea Nitrogen (mmol/L):", min_value=0, max_value=1000, value=0)
SP = st.sidebar.number_input("Serum Potassium (mmol/L):", min_value=0, max_value=1000, value=0)
UA = st.sidebar.number_input("Uric Acid (μmol/L):", min_value=0, max_value=1000, value=0)
FA = st.sidebar.number_input("Folic Acid (nmol/L):", min_value=0, max_value=1000, value=0)
V12 = st.sidebar.number_input("VitB12 (pmol/L):", min_value=0, max_value=2000, value=0)

# Total Iron-Binding Capacity (umol/l) input
TIBC = st.sidebar.number_input("TIBC (umol/l):", min_value=0, max_value=1000, value=45)

# Serum Ferritin (ng/ml),Soluble Transferrin Receptor (mg/L)
SF = st.sidebar.number_input("Serum Ferritin (ng/ml):", min_value=0, max_value=1000, value=0)
STR = st.sidebar.number_input("Soluble Transferrin Receptor (mg/L):", min_value=0, max_value=1000, value=0)

# Nutritional Support Therapy input
NST = st.sidebar.selectbox("Nutritional Support Therapy:", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 0 else 'No (0)')

# Process the input and make a prediction
feature_values = [cod, dur, AOO, Smoke, ACH, GSH, Dia, GB, Vom, ABCH, ANCH, FOLS, EIM, CDAI, ACT, 
                  RBC, HB, MCV, CRP, TP, UN, SP, UA, FA, V12, TIBC, SF, STR, NST]
features = np.array([feature_values])  # 转换为NumPy数组

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = model.predict(features)[0]  # 预测Task 1类别
    predicted_proba = model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class:** {predicted_class}")  # 显示预测的类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

    if predicted_class == 1:  # 如果预测一年内发生并发症
        advice = (
            f"According to our model, the patient is classified as high risk for anal complications (ulcer, fistula, or abscess) within 12 months. "
            f"The estimated probability of anal complications over the next year is {probability:.1f}%."
            "This indicates an increased likelihood of clinically meaningful perianal events and may warrant earlier proactive management. "
            "The prediction is intended for risk stratification and decision support and should be interpreted alongside clinical examination and objective disease activity measures. "
            "Given that Task 2 risk is predominantly associated with prior perianal history, extra-intestinal manifestations, and higher disease activity, we recommend closer surveillance with timely reassessment of perianal disease status, optimization/escalation of anti-inflammatory therapy when indicated, and consideration of early multidisciplinary input (IBD specialist + colorectal surgery/radiology) for suspected fistulizing/abscess disease. "
            "Urgent evaluation is warranted if there are features suggestive of perianal abscess or sepsis (e.g., rapidly progressive pain, fever, fluctuance, systemic toxicity)."
        )  # 如果预测会发生并发症，给出相关建议
    else:  # 如果预测不会
        advice = (
            f"According to our model, the patient is classified as low risk for anal complications (ulcer, fistula, or abscess) within 12 months."
            f"The estimated probability of no anal complications over the next year is {probability:.1f}%."
            "This supports a lower short-term risk of perianal events under the current clinical context; however, perianal CD is characteristically relapsing-remitting and risk may change with fluctuations in systemic inflammatory burden. "
            "In our framework, task-specific risk is primarily driven by prior perianal complication history, extra-intestinal manifestations, and clinical disease activity (e.g., CDAI), which should be integrated with objective assessment."
            "We recommend continuing standard follow-up, with reassessment triggered by rising inflammatory activity, new or worsening perianal symptoms, or changes in systemic disease behavior." 
        )

    st.write(advice)  # 显示建议

    # Visualize the prediction probabilities
    sample_prob = {
        'Class_0': predicted_proba[0],  # 类别0的概率
        'Class_1': predicted_proba[1]  # 类别1的概率
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # 设置图形大小

    # Create bar chart
    bars = plt.barh(['No Anal Comp', 'Yes'], 
                    [sample_prob['Class_0'], sample_prob['Class_1']], 
                    color=['#512b58', '#fe346e'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for Patient", fontsize=20, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=14, fontweight='bold')  # 添加X轴标签，并设置字体大小和加粗
    plt.ylabel("Classes", fontsize=14, fontweight='bold')  # 添加Y轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['Class_0'], sample_prob['Class_1']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot

    st.pyplot(plt)  # 显示图表



