
# Load data
# Note: Ensure '12交集特征.xlsx' is in the same directory or provide the full path
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
# Note: Ensure '12交集特征.xlsx' is in the same directory or provide the full path
try:
    df = pd.read_excel('12交集特征.xlsx')
except FileNotFoundError:
    st.error("Error, file not found")
    st.stop()
df.rename(columns={"年龄":"年龄",
                   "BMI":"BMI",
                   "LAC_D1":"LAC_D1",
                   "PO2/FiO2_D2":"PO2/FiO2_D2",
                   "24小时总尿量(ml)":"24小时总尿量(ml)",
                   "住ICU时间":"住ICU时间",
                   "APACHE Ⅱ":"APACHE Ⅱ",
                   "SOFA":"SOFA",
                   "镇静时间(hr)":"镇静时间(hr)",
                   "ARDS分级":"ARDS分级是否为3级",
                   "免疫抑制人群":"是否为免疫抑制人群",
                   "呼吸支持方式":"呼吸支持方式是否为机械通气"},inplace=True)
#删除rename
# Define variables
continuous_vars = [
'年龄',
'BMI',
'LAC_D1',
'PO2/FiO2_D2',
'24小时总尿量(ml)',
'住ICU时间',
'APACHE Ⅱ',
'SOFA',
'镇静时间(hr)']
categorical_vars = [
    'ARDS分级是否为3级',  # 使用重命名后的列名
    '是否为免疫抑制人群',  # 使用重命名后的列名
    '呼吸支持方式是否为机械通气',  # 使用重命名后的列名
]
# Combine all variables for unified input
all_vars = continuous_vars + categorical_vars
# 预处理管道，对分类变量进行OneHotEncoder（不删除任何列）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_vars),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_vars)  # 这里不再传递selected_categorical_vars，而是使用categorical_vars，并且OneHotEncoder不传递参数
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
drop_columns = ['是否为免疫抑制人群_0', 'ARDS分级是否为3级_1', 'ARDS分级是否为3级_2', '呼吸支持方式是否为机械通气_1', '呼吸支持方式是否为机械通气_2']

# 只删除存在的列
columns_to_drop = [col for col in drop_columns if col in X_processed_df.columns]
X_processed_df = X_processed_df.drop(columns=columns_to_drop)

# 然后，您可以使用X_processed_df作为特征

X = X_processed_df
y = df['结局']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)


# --- Streamlit App Interface ---

# Centered Title
st.markdown("<h1 style='text-align: center;'>Prognostic model of ARDS patients based on logistic regression</h1>", unsafe_allow_html=True)

# --- 1. User Input for X values (Unified, 4 columns) ---
st.header("1. Enter Patient Data")
user_input = {} # summarize user input data
input_valid = True # Flag to check if all inputs are valid
# Create input fields for all variables in 4 columns
# Combine continuous and categorical for unified handling in layout
input_cols = st.columns(4) # Changed to 4 columns
for i, var in enumerate(all_vars):
    with input_cols[i % 4]: # Cycle through 4 columns
        if var in continuous_vars:
            # Handle continuous variables - No default value
            # 修正：移除FiO2的特殊处理，因为变量列表中只有PO2/FiO2_D2
            user_val = st.number_input(f"{var}", value=None, format="%.4f", step=0.01, placeholder="please enter")
            if user_val is None:
                input_valid = False
                #st.warning(f"请输入 {var} 的值")
            user_input[var] = user_val
        else: # Handle categorical variables
            # 修正：直接从数据中获取唯一值，而不是从编码器中获取
            options = np.unique(df[var].astype(str))
            # Default to the first category or a placeholder
            # UI - No default selection, user must choose
            selected_option = st.selectbox(f"{var}", options=options, index=None, placeholder="please enter")
            if selected_option is None:
                input_valid = False
                #st.warning(f"请选择 {var} 的值")
            user_input[var] = selected_option


# --- 3. Prediction Button and Logic ---
if st.button("Train Model and Predict"):
    if not input_valid:
        st.error("error, please check all X is inputed")
    else:
        
        # --- Train the model with fixed parameters ---
        try:
             # 创建输入数据的DataFrame
            input_data = pd.DataFrame([user_input])
            
            # 应用相同的预处理
            input_processed = preprocessor.transform(input_data)
            input_processed_df = pd.DataFrame(input_processed, columns=feature_names)
            
            # 确保列的顺序与训练时完全一致
            input_processed_df = input_processed_df[training_feature_columns]
            
            # 训练模型
            model = LogisticRegression(
                random_state=999,
                penalty='l2',
                C=0.3593813663804626
            )
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
**Disclaimer:**

Supplement:
*   P02/FIO2_D2代表ARDS诊断第二天的氧合指数。
*   APACHEII和SOFA评分为ARDS诊断当天的评分。
*   LAC_D1代表ARDS诊断当天的乳酸水平。
*   呼吸支持方式是否为有创机械通气：1代表有创机械通气；0代表未进行有创机械通气。
*   是否为免疫抑制人群：1代表长期激素治疗（等效泼尼松≥20mg/d持续≥14天或总剂量＞700mg）; 0则无长期激素治疗。
*   24小时总尿量代表着ARDS诊断后24小时的总尿量。
*   ARDS分级是否为3级，1代表确诊ARDS的氧合指数小于等于100，0代表确诊ARDS的氧合指数大于100。
*   住ICU时间单位为小时，一天即24H。

"""
st.markdown(disclaimer_text)





