# 🛡️ TensorVeil: Privacy-Preserving Synthetic Data Engine

TensorVeil is a machine learning application designed to generate high-quality synthetic data that preserves the statistical properties of the original dataset while protecting user privacy. Powered by **CTGAN (Conditional Tabular GANs)**, it allows users to train models on sensitive data and export safe, synthetic replicas.

## 🚀 Key Features
* **Automated Analysis:** instantly scans datasets to detect categorical vs. continuous variables.
* **Privacy Engine:** Uses deep learning (CTGAN) to learn hidden correlations without memorizing exact records.
* **Smart Pre-processing:** Auto-cleans missing values and handles data formatting.
* **Interactive UI:** A user-friendly Streamlit interface for non-technical users.
* **Modular Architecture:** Built with a scalable, maintainable codebase.

## 🏗️ Project Architecture
The project has been refactored into a modular design for scalability:

| Module | Responsibility |
| :--- | :--- |
| **`app.py`** | **The Interface:** Handles the UI/UX, user inputs, and workflow management. |
| **`analyzer.py`** | **The Eyes:** Scans raw data to identify data types and schema structure. |
| **`generator.py`** | **The Brain:** Encapsulates the CTGAN model, training logic, and data generation. |

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Core Logic:** CTGAN (SDV), PyTorch, Pandas
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
2. **Train** : Go to Tab 2, set your Epochs (recommended: 50+), and click "Start Training".
3. **Export** : Go to Tab 3 to download your privacy-preserved synthetic dataset.