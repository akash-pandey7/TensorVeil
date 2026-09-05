# 🛡️ TensorVeil: Privacy-Preserving Synthetic Data Engine

TensorVeil is a machine learning application designed to generate high-quality synthetic data that preserves the statistical properties of the original dataset while protecting user privacy. Powered by **CTGAN (Conditional Tabular GANs)**, it allows users to train models on sensitive data, evaluate the result with a built-in quality/utility/privacy metrics suite, and export safe, synthetic replicas.

## Key Features

* **Automated Analysis:** Instantly scans datasets to detect categorical vs. continuous variables.
* **Privacy Engine:** Uses deep learning (CTGAN) to learn hidden correlations without memorizing exact records.
* **Configurable Training:** Exposes CTGAN's core hyperparameters (epochs, generator/discriminator network size, PAC, batch size) so training can be tuned per dataset rather than relying on one fixed configuration.
* **Quality Evaluation Suite:** Goes beyond visual inspection with quantitative metrics:
  * **Statistical Fidelity** — per-column KS-test (continuous) and Total Variation Distance (categorical), rolled into a single similarity score.
  * **Correlation Preservation** — compares real vs. synthetic Pearson correlation matrices.
  * **Privacy (DCR)** — Distance to Closest Record, measuring how far synthetic rows sit from real ones.
  * **Downstream Utility (TSTR/TRTR)** — trains a classifier/regressor on synthetic vs. real data and compares accuracy on a shared, held-out real test set, directly answering "is this synthetic data actually useful for machine learning."
* **Quality Inspection Dashboard:** Visual real-vs-synthetic comparisons (histograms, bar charts) alongside the quantitative metrics above.
* **Smart Pre-processing:** Auto-cleans missing values and handles data formatting.
* **Optional Experiment History:** Training runs can be logged to a Supabase-backed history table; the core generation and evaluation workflow works fully offline without a database connection.
* **Modular Architecture:** Built with a scalable, maintainable codebase, with automated tests and CI.

## 🌐 Live Demo

**[Insert your Streamlit Cloud Link Here]**

## Project Architecture

The project follows a modular design for scalability:

| Module | Responsibility |
| :--- | :--- |
| **`app.py`** | **The Interface:** Handles the UI/UX, user inputs, and workflow management across Upload, Train, Metrics/Export, and History tabs. |
| **`analyzer.py`** | **The Eyes:** Scans raw data to identify categorical vs. continuous columns. |
| **`generator.py`** | **The Brain:** Encapsulates the CTGAN model, configurable training, and data generation. |
| **`metrics.py`** | **The Judge:** Evaluates synthetic data quality — statistical similarity, correlation preservation, privacy (DCR), and downstream utility (TSTR/TRTR). |

## Tech Stack

* **Language:** Python 3.x
* **Core Logic:** CTGAN (SDV), Pandas, Numpy
* **Evaluation:** scikit-learn (Random Forest, NearestNeighbors, preprocessing pipelines), SciPy (KS-test)
* **Visualization:** Matplotlib, Streamlit Charts
* **Interface:** Streamlit
* **Persistence (optional):** Supabase
* **CI:** GitHub Actions

## Installation & Usage

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

## Workflow

1. **Upload:** Drop your CSV/Excel file in Tab 1. The system will auto-analyze the schema.
2. **Train:** Go to Tab 2, set Epochs and (optionally) advanced CTGAN settings, and click "Start Training."
3. **Evaluate:** Switch to Tab 3, select a target column, and click "Calculate Metrics" to see statistical similarity, correlation preservation, privacy (DCR), and utility (TSTR/TRTR) scores — alongside the real-vs-synthetic visual comparison.
4. **Export:** Download your privacy-preserved synthetic dataset as a CSV.

> **Note on training time:** For datasets with a large row count, training can take 20–30 minutes to produce a good-quality synthetic dataset. For larger datasets specifically, use a minimum of 100 epochs — CTGAN tends to converge well with fewer epochs on bigger datasets, so 100+ is usually enough rather than needing the higher epoch counts smaller datasets may require.

## Findings

Testing across four datasets of varying size and structure revealed that the right training configuration depends heavily on the dataset itself — there's no single epoch count that works best everywhere:

| Dataset | Rows | Best Epochs Found | TSTR / TRTR Gap | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Titanic | ~890 | ~250 (plateaus early) | ~0.11 | Mostly categorical features; accuracy gap stayed flat from 250 to 750 epochs — additional training gave no benefit. |
| Wine Quality | ~1,100 | 750 | ~0.18 | Strongly epoch-sensitive: gap fell from 0.46 (250 epochs) to 0.18 (750 epochs), then degraded at 1,000 epochs — a clear overfitting point where privacy (DCR) also declined. |
| Adult Income | ~45,000 | 100 | ~0.03 | Largest dataset tested; converged to near-parity between synthetic and real training accuracy in relatively few epochs. |
| Iris | 150 | — | Unreliable | Too few rows for a stable held-out test split; TSTR/TRTR accuracy varied by several points between identical reruns. Statistical similarity, correlation, and DCR metrics remained meaningful. |

**Takeaways:**
* Larger, simpler (more categorical) datasets tend to converge quickly and are less sensitive to epoch count.
* Smaller datasets with complex, continuous, correlated features (like wine chemistry) benefit substantially from more training — but only up to a point; excessive training reduced both utility and the synthetic data's distance from real records.
* Increasing CTGAN's network capacity (`generator_dim`/`discriminator_dim`) did **not** help wine quality's accuracy gap at a fixed epoch budget — it was epoch count, not model size, that mattered.

## ⚠️ Known Limitations

* **Utility metrics on small datasets:** TSTR/TRTR needs a held-out real test split to compare against, and on datasets under a few hundred rows that split is too small to give a stable reading — accuracy can swing several points between identical runs, as seen on Iris. Statistical similarity, correlation, and DCR don't have this problem and stay reliable regardless of dataset size.
* **No formal privacy guarantee:** DCR is an empirical distance-based privacy signal, not a formal guarantee like differential privacy. It's one useful check, not a certification that the data is anonymous.
* **CTGAN's adversarial training can be unstable:**  The loss curves show real oscillation during training. More epochs isn't always better, which is exactly what the wine quality results above show.
