import streamlit as st
import pandas as pd

from src.analyzer import analyze_data
from src.generator import TensorVeilGenerator

# CONFIGURATION
st.set_page_config(page_title = "TensorVeil", page_icon = "🛡️", layout = "wide")
st.title("🛡️ TensorVeil : Synthetic Data Engine")

# SESSION STATE SETUP
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'categorical_columns' not in st.session_state:
    st.session_state['categorical_columns'] = None
if 'synthetic_data' not in st.session_state:
    st.session_state['synthetic_data'] = None
# We must store the trained model or it will get deleted on refresh
if 'generator_model' not in st.session_state:
    st.session_state['generator_model'] = None
    

# UI TABS
tab1, tab2, tab3 = st.tabs(["📂 1. Upload", "⚙️ 2. Train", "📥 3. Export"])

# TAB1
with tab1:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type = ["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith("csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Auto clean the null value rows
            if df.isnull().sum().sum() > 0:
                st.warning("Found empty cells. Removing missing value rows...")
                df = df.dropna()
                st.success("Cleaned missing values.")
            st.session_state['df'] = df
            
            # CALL ANALYZER MODULE
            cat_cols = analyze_data(df)
            st.session_state['categorical_columns'] = cat_cols
            
            st.success("Data Loaded!")
            st.info(f"Analysis : Found {len(cat_cols)} categorical_columns.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error : {e}")

# TAB2
with tab2:
    st.header("Train Model")
    if st.session_state['df'] is not None:
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs", min_value = 1, value = 50, step = 10)
        with col2:
            count = st.number_input("Count", min_value = 1, value = 100)\
        
        if st.button("🚀 Start Training"):
            st.write("Initializing Engine...")
            
            # CALL GENERATOR MODULE
            gen = TensorVeilGenerator(epochs=epochs)
            
            with st.spinner("Training..."):
                gen.train(st.session_state['df'], st.session_state['categorical_columns'])
            
            # Save the trained engine to memory
            st.session_state['generator_model'] = gen
            st.success("Training Complete!")
            
            # Generate Data
            with st.spinner("Generating..."):
                new_data = gen.generate(count)
                st.session_state['synthetic_data'] = new_data
            
            st.success(f"Generated {len(new_data)} rows!")
            st.dataframe(new_data.head())
        else:
            st.warning("Please upload the data in Tab 1 first.")

# TAB3
with tab3:
    st.header("Download")
    
    if st.session_state["synthetic_data"] is not None:
        csv = st.session_state["synthetic_data"].to_csv(index = False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="tensorveil_synthetic.csv",
            mime="text/csv"
        )
    else:
        st.info("Generate data in Tab 2 first.")