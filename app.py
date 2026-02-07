import matplotlib.pyplot as plt
import streamlit as st # type: ignore
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
    st.header("Quality Inspection & Export")
    
    if st.session_state["synthetic_data"] is not None:
        # 1. Selector
        selected_col = st.selectbox(
            label="Select Column to Compare", 
            options=st.session_state["synthetic_data"].columns
        )
        
        # 2. Vital Stats (Side-by-Side)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Real Data Unique Values", 
                value=st.session_state['df'][selected_col].nunique()
            )
        with col2:
            st.metric(
                label="Synthetic Data Unique Values", 
                value=st.session_state['synthetic_data'][selected_col].nunique()
            )
            
        # 3. Determine Data Type
        # We check if the Real Data column is a number (int or float)
        is_numeric = pd.api.types.is_numeric_dtype(st.session_state['df'][selected_col])
        
        # 4. Plotting Logic
        if is_numeric:
            st.subheader(f"Distribution of {selected_col}")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Plot Real Data as a solid, light color
            ax.hist(st.session_state['df'][selected_col], bins=20, density=True, label="Real", alpha=0.5, color='blue')
            
            # Plot Synthetic Data as a thick, dark OUTLINE (step)
            # This makes it easy to see "through" the data
            ax.hist(st.session_state['synthetic_data'][selected_col], bins=20, density=True, label="Synthetic", histtype='step', linewidth=2, color='black')
            
            ax.set_title("Real (Blue) vs. Synthetic (Black Outline)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.subheader(f"Count of {selected_col}")
            # Fix: Streamlit's built-in bar chart is safer for text categories than Matplotlib
            real_counts = st.session_state['df'][selected_col].value_counts()
            syn_counts = st.session_state['synthetic_data'][selected_col].value_counts()
            
            # Combine into a clean table for plotting
            chart_data = pd.DataFrame({
                "Real": real_counts,
                "Synthetic": syn_counts
            })
            st.bar_chart(chart_data)

        # 5. Download Button
        st.divider() # Adds a nice line separator
        csv = st.session_state["synthetic_data"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Synthetic CSV",
            data=csv,
            file_name="tensorveil_synthetic.csv",
            mime="text/csv"
        )
    else:
        st.info("⚠️ Please generate data in Tab 2 first.")