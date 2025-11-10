
# Load data
# Note: Ensure '20交集特征.xlsx' is in the same directory or provide the full path
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# --- Set wide layout ---
st.set_page_config(layout="wide")

# --- Your Provided Code (Adapted for Streamlit) ---

# Load data
# Note: Ensure '20交集特征.xlsx' is in the same directory or provide the full path
try:
    df = pd.read_excel('交集特征20.xlsx')
except FileNotFoundError:
    st.error("Error, file not found")
    st.stop()
df.rename(columns={
                    "APACHE Ⅱ":"APACHE Ⅱ",
                    "SOFA":"SOFA",
                    "LAC_D1":"LAC(mmol/L)_D1",
                    "PO2/FiO2_D2":"PO2/FiO2_D2",
                    "BMI":"BMI",
                    "WBC":"WBC(*10^9/L)_D1",
                    "LYM"："LYM(*10^9/L)_D1",
                    "PH_D2"："PH_D2",
                    "镇静时间(hr)"："镇静时间(hr)",
                    "24小时总尿量(ml)"："24小时总尿量(ml)",
                    "吸烟指数"："吸烟指数",
                    "TBIL(μmolL)"："TBIL(μmolL)_D1",
                    "PO2_D1"："PO2(mmHg)_D1",
                    "ARDS分级":"ARDS分级",
                    "免疫抑制人群":"是否为免疫抑制人群",
                    "冠心病":"是否患有冠心病"},inplace=True)
#删除rename
# Define variables
continuous_vars = [
'APACHE Ⅱ',
'SOFA',
'LAC(mmol/L)_D1',
'PO2/FiO2_D2',
'BMI',
'WBC(*10^9/L)_D1',
'LYM(*10^9/L)_D1',
'PH_D2',
'镇静时间(hr)',
'24小时总尿量(ml)',
'吸烟指数',
'TBIL(μmolL)_D1',
'PO2(mmHg)_D1',



]
categorical_vars = [
'ARDS分级', # 使用重命名后的列名
'是否为免疫抑制人群',
'是否患有冠心病'
]
# Combine all variables for unified input
all_vars = continuous_vars + categorical_vars
# 预处理管道，对分类变量进行OneHotEncoder（不删除任何列）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_vars),
        ('cat', OneHotEncoder(drop='first',handle_unknown='ignore'), categorical_vars)  # 这里不再传递selected_categorical_vars，而是使用categorical_vars，并且OneHotEncoder不传递参数
    ])

# 应用预处理
X_processed = preprocessor.fit_transform(df)

# 获取特征名
try:
    feature_names = (
        continuous_vars +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_vars))
    )
except AttributeError:
    feature_names = (
        continuous_vars +
        list(preprocessor.named_transformers_['cat'].get_feature_names(categorical_vars))
    )

X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# 定义要删除的列
drop_columns = ['ARDS分级_1', 'ARDS分级_2']

# 只删除存在的列
columns_to_drop = [col for col in drop_columns if col in X_processed_df.columns]
X_processed_df = X_processed_df.drop(columns=columns_to_drop)

# 然后，您可以使用X_processed_df作为特征

X = X_processed_df
y = df['结局']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

# 保存训练时的特征列顺序
training_feature_columns = X_train.columns.tolist()

# --- Streamlit App Interface ---

st.markdown("<h1 style='text-align: center;'>基于支持向量机的急性呼吸窘迫综合征患者的预后模型</h1>", unsafe_allow_html=True)

# --- 1. User Input for X values ---
st.header("1. 请输入患者信息")
user_input = {}
input_valid = True

input_cols = st.columns(4)
for i, var in enumerate(all_vars):
    with input_cols[i % 4]:
        if var in continuous_vars:
            user_val = st.number_input(f"{var}", value=None, format="%.4f", step=0.01, placeholder="please enter")
            if user_val is None:
                input_valid = False
            user_input[var] = user_val
        else:
            options = np.unique(df[var].astype(str))
            selected_option = st.selectbox(f"{var}", options=options, index=None, placeholder="please enter")
            if selected_option is None:
                input_valid = False
            user_input[var] = selected_option

# --- 3. Prediction Button and Logic ---
if st.button("预测患者死亡概率为"):
    if not input_valid:
        st.error("error, please check all X is inputed")
    else:
        try:
            # 创建输入数据的DataFrame
            input_data = pd.DataFrame([user_input])
            
            # 应用相同的预处理
            input_processed = preprocessor.transform(input_data)
            input_processed_df = pd.DataFrame(input_processed, columns=feature_names)
            
            # 确保列的顺序与训练时完全一致
            input_processed_df = input_processed_df[training_feature_columns]
            
            # 训练模型
            model = SVC(random_state=999,
                        kernel='rbf',
                        probability=True,
                        C = 0.5,
                        gamma=0.01)
            model.fit(X_train, y_train)
            st.success("Model trained successfully with fixed parameters!")
            
            # 进行预测
            prediction_proba = model.predict_proba(input_processed_df)[0]
            
            # 显示结果
            st.header("Prediction Result")
            prob_label = "Mortality probability of ARDS"
            st.metric(label=prob_label, value=f"{prediction_proba[1]*100:.2f}%")
            
        except Exception as e:
            st.error(f"An error occurred during model training or prediction: {e}")
            # 添加调试信息
            st.write(f"Training features: {len(training_feature_columns)}")
            st.write(f"Input features after preprocessing: {len(input_processed_df.columns)}")
            st.write(f"Training feature columns: {training_feature_columns}")
            st.write(f"Input feature columns: {input_processed_df.columns.tolist()}")

# --- Disclaimer Section at the Bottom ---
st.markdown("---")
disclaimer_text = """
**补充说明:**
*   D1、D2分别代表诊断ARDS第一天和第二天。
*   APACHEII和SOFA评分为ARDS诊断当天的评分。
*   是否为免疫抑制人群:1代表长期激素治疗（等效泼尼松≥20mg/d持续≥14天或总剂量＞700mg）; 0则无长期激素治疗。
*   24小时总尿量代表着ARDS诊断后24小时的总尿量。
*   ARDS分级：根据柏林定义，1代表轻度，2代表中度，3代表重度。
*   吸烟指数=每日吸烟指数*吸烟年数
"""
st.markdown(disclaimer_text)



