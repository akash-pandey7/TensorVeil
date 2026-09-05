import matplotlib.pyplot as plt
import streamlit as st # type: ignore
import pandas as pd

from st_supabase_connection import SupabaseConnection
from src.analyzer import analyze_data
from src.generator import TensorVeilGenerator
from src.metrics import aggregate_metrics

# CONFIGURATION
st.set_page_config(page_title = "TensorVeil", page_icon = "🛡️", layout = "wide")
st.title("🛡️ TensorVeil : Synthetic Data Engine")

# Database Connection
try:
    supabase_url = st.secrets["supabase"]["url"]
    supabase_key = st.secrets["supabase"]["key"]
    conn = st.connection(
    "supabase",
    type=SupabaseConnection,
    url = supabase_url,
    key = supabase_key
    )
except FileNotFoundError:
    st.error("❌ secrets.toml file not found.")
    conn = None
except KeyError:
    st.error("❌ Secrets found, but [supabase] section or keys are missing.")
    conn = None

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
    
if "user" not in st.session_state:
    st.session_state["user"] = None

with st.sidebar:
    st.subheader("Account")

    if st.session_state["user"] is not None:
        st.success(f"Logged in as {st.session_state['user'].email}")
        if st.button("Log Out"):
            st.session_state["user"] = None
            st.rerun()
    else:
        auth_mode = st.radio(
            "Access",
            options=["Continue without login", "Log In", "Sign Up"],
            index=0,
        )

        if auth_mode == "Log In":
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log In")
                if submitted:
                    if conn is None:
                        st.error("❌ Login unavailable — no database connection.")
                    else:
                        try:
                            response = conn.auth.sign_in_with_password(
                                {"email": email, "password": password}
                            )
                            st.session_state["user"] = response.user
                            st.success("Logged in!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Login failed: {e}")

        elif auth_mode == "Sign Up":
            with st.form("signup_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign Up")
                if submitted:
                    if conn is None:
                        st.error("❌ Sign up unavailable — no database connection.")
                    else:
                        try:
                            response = conn.auth.sign_up(
                                {"email": email, "password": password}
                            )
                            if response.user is not None:
                                st.success(
                                    "Account created! Check your email to confirm, "
                                    "then log in above."
                                )
                            else:
                                st.warning("Sign up did not return a user — check your Supabase Auth settings.")
                        except Exception as e:
                            st.error(f"Sign up failed: {e}")

        else:
            st.caption("Using TensorVeil as a guest — your history won't be saved.")

# UI TABS
tab1, tab2, tab3, tab4 = st.tabs(["📂 1. Upload", "⚙️ 2. Train", "📥 3. Export", "📜 History"])

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
            st.session_state["uploaded_file_name"] = uploaded_file.name
            # Auto clean the null value rows
            df = df.replace("?", pd.NA)
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
            epochs = st.number_input("Epochs", min_value = 1, value = 250, step = 5)
        with col2:
            count = st.number_input("Count", min_value = 1, value = 100)
        
        if st.button("🚀 Start Training"):
            st.write("Initializing Engine...")
            gen = TensorVeilGenerator(epochs=epochs, generator_dim=(256, 256), discriminator_dim=(256, 256), pac=10, batch_size=500)

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                gen.train(
                    st.session_state['df'],
                    st.session_state['categorical_columns'],
                    progress_bar=progress_bar,
                    status_text=status_text
                )
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()

            st.session_state['generator_model'] = gen
            st.success("Model is ready")

            # Show real loss curve from CTGAN
            loss_df = gen.get_loss_history()
            if not loss_df.empty:
                st.subheader("📉 Training Loss")
                st.line_chart(loss_df.set_index('Epoch')[['Generator Loss', 'Discriminator Loss']])

            # Save to database
            if st.session_state["user"] is not None and conn is not None:
                try:
                    with st.spinner("Saving experiment to history..."):
                        conn.table("experiments").insert({
                            "user_id": st.session_state["user"].id,
                            "dataset_name": st.session_state['uploaded_file_name'],
                            "epochs": epochs,
                            "row_count": len(st.session_state['df']),
                            "status": "completed"
                        }).execute()
                        st.toast("✅ Experiment saved to your history!", icon="☁️")
                except Exception as e:
                    st.error(f"⚠️ Could not save to database: {e}")
            elif st.session_state["user"] is None:
                st.caption("Log in to save this run to your history.")

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
            # 4. Target Column Selector for Metrics
            selected_target = st.selectbox(
                label="Select Target Column for Metrics",
                options=st.session_state["synthetic_data"].columns
            )
            
            if st.button("📊 Calculate Metrics"):
                with st.spinner("Calculating metrics..."):
                    metrics = aggregate_metrics(
                        st.session_state['df'],
                        st.session_state['synthetic_data'],
                        target_column=selected_target,
                        task = "classification" if selected_target in st.session_state['categorical_columns'] else "regression"
                    )
                st.success("Metrics Calculated!")
                st.metric("Mean Statistical Similarity", f"{metrics['statistical_similarity']['mean_similarity']:.2f}")
                st.metric("Mean Absolute Correlation Difference", f"{metrics['correlation']['mean_absolute_difference']:.2f}")
                st.metric("Mean DCR (Distance to Closest Record)", f"{metrics['dcr']['median']:.2f}")
    
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("TSTR Accuracy", f"{metrics['utility']['tstr']['accuracy']:.2f}")
                with col2:
                    st.metric("TRTR Accuracy", f"{metrics['utility']['trtr']['accuracy']:.2f}")
    
                with st.expander("View Full Metrics", expanded=False):
                    st.json(metrics)
                    
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

# TAB4
with tab4:
    st.header("📜 Training History")

    if st.session_state["user"] is None:
        st.info("🔒 Log in from the sidebar to view your saved experiment history.")
    elif conn is None:
        st.warning("⚠️ History unavailable — no database connection.")
    else:
        # Fetch data from database, scoped to the logged-in user
        try:
            response = (
                conn.table("experiments")
                .select("*")
                .eq("user_id", st.session_state["user"].id)
                .execute()
            )

            # Check if data exists
            if response.data:
                history_df = pd.DataFrame(response.data)

                if "created_at" in history_df.columns:
                    history_df["created_at"] = pd.to_datetime(history_df["created_at"])
                    history_df = history_df.sort_values(by="created_at", ascending=False)

                display_cols = history_df[["created_at", "dataset_name", "epochs", "row_count", "status"]]
                available_cols = [c for c in display_cols if c in history_df.columns]

                st.dataframe(
                    display_cols[available_cols],
                    column_config={
                        "created_at": st.column_config.DatetimeColumn("Date", format="D MMM YYYY, h:mm a"),
                        "dataset_name": "Dataset",
                        "epochs": "Epochs",
                        "row_count": "Row Count",
                        "status": st.column_config.TextColumn("Status", help="Training status")
                    },
                    width="stretch",
                    hide_index=True
                )
            else:
                st.info("No training history found. Run some new experiments")
        except Exception as e:
            st.error(f"❌ Error fetching history: {e}")