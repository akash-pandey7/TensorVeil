# 🛡️ TensorVeil: Privacy-Preserving Synthetic Data Engine

TensorVeil is a machine learning application designed to generate high-quality synthetic data that preserves the statistical properties of the original dataset while protecting user privacy. Powered by **CTGAN (Conditional Tabular GANs)**, it allows users to train models on sensitive data and export safe, synthetic replicas.

## 🚀 Key Features

* **Automated Analysis:** Instantly scans datasets to detect categorical vs. continuous variables.
* **Privacy Engine:** Uses deep learning (CTGAN) to learn hidden correlations without memorizing exact records.
* **Smart Pre-processing:** Auto-cleans missing values and handles data formatting.
* **Quality Inspection Dashboard:** Built-in visualization tools (Histograms & Bar Charts) to compare real vs. synthetic data distributions side-by-side.
* **Modular Architecture:** Built with a scalable, maintainable codebase.

## 🌐 Live Demo

**[Insert your Streamlit Cloud Link Here]**

## 🏗️ Project Architecture

The project follows a modular design for scalability:

| Module | Responsibility |
| :--- | :--- |
| **`app.py`** | **The Interface:** Handles the UI/UX, user inputs, and workflow management. |
| **`analyzer.py`** | **The Eyes:** Scans raw data to identify data types and schema structure. |
| **`generator.py`** | **The Brain:** Encapsulates the CTGAN model, training logic, and data generation. |

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Core Logic:** CTGAN (SDV), PyTorch, Pandas
* **Visualization:** Matplotlib, Streamlit Charts
* **Interface:** Streamlit
* **Utils:** Joblib (for threading optimization)

## 💻 Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/akash-pandey7/TensorVeil.git
cd TensorVeil
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the application**
```bash
streamlit run app.py
```

## 📝 Workflow
1. **Upload** : Drop your CSV/Excel file in Tab 1. The system will auto-analyze the schema.
2. **Train** : Go to Tab 2, set your Epochs (recommended: 100+), and click "Start Training".
3. **Validate** : Switch to Tab 3 to visually compare the "Real" vs. "Synthetic" data distributions.
4. **Export**: Download your privacy-preserved synthetic dataset.