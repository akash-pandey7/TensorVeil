# 🛡️ [TensorVeil](https://tensorveil.streamlit.app/): Privacy-Preserving Synthetic Data Engine

TensorVeil is a machine learning application designed to generate high-quality synthetic data that preserves the statistical properties of the original dataset while protecting user privacy. Powered by **CTGAN (Conditional Tabular GANs)**, it allows users to train models on sensitive data, evaluate the result with a built-in quality/utility/privacy metrics suite, and export safe, synthetic replicas.

## Key Features

* **Automated Analysis:** Instantly scans datasets to detect categorical vs. continuous variables.
* **Privacy Engine:** Uses deep learning (CTGAN) to learn hidden correlations without memorizing exact records.
* **Configurable Training:** Exposes CTGAN's core hyperparameters (epochs, generator/discriminator network size, PAC, batch size) so training can be tuned per dataset rather than relying on one fixed configuration. Values are capped (e.g. epochs at 1000, network width at 1024) since this app runs on shared, CPU-only hosting with no per-user resource isolation.
* **Quality Evaluation Suite:** Goes beyond visual inspection with quantitative metrics:
  * **Statistical Fidelity** - per-column KS-test (continuous) and Total Variation Distance (categorical), rolled into a single similarity score.
  * **Correlation Preservation** - compares real vs. synthetic Pearson correlation matrices.
  * **Privacy (DCR)** - Distance to Closest Record, benchmarked against real-to-real row spacing to flag potential memorization - not just a raw distance, but a ratio you can actually interpret.
  * **Downstream Utility (TSTR/TRTR)** - trains a classifier/regressor on synthetic vs. real data and compares accuracy on a shared, held-out real test set, directly answering "is this synthetic data actually useful for machine learning."
* **Quality Inspection Dashboard:** Visual real-vs-synthetic comparisons (histograms, bar charts) alongside the quantitative metrics above.
* **Smart Pre-processing:** Imputes missing values (median for numeric columns, mode for categorical) and drops columns that are mostly empty, rather than silently discarding rows.
* **Optional Experiment History:** Training runs can be logged to a Supabase-backed history table; the core generation and evaluation workflow works fully offline without a database connection.
* **Modular Architecture:** Built with a scalable, maintainable codebase, with automated tests and CI.

## 🌐 Live Demo : **[TensorVeil](https://tensorveil.streamlit.app/)**
>
> **Note:** Demo may take a moment to wake up if it's been idle.

## Project Architecture

The project follows a modular design for scalability:

| Module | Responsibility |
| :--- | :--- |
| **`app.py`** | **The Interface:** Handles the UI/UX, user inputs, and workflow management across Upload, Train, Metrics/Export, and History tabs. |
| **`analyzer.py`** | **The Eyes:** Scans raw data to identify categorical vs. continuous columns. |
| **`generator.py`** | **The Brain:** Encapsulates the CTGAN model, configurable training, and data generation. |
| **`metrics.py`** | **The Judge:** Evaluates synthetic data quality - statistical similarity, correlation preservation, privacy (DCR), and downstream utility (TSTR/TRTR). |

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
3. **Evaluate:** Switch to Tab 3, select a target column, and click "Calculate Metrics" to see statistical similarity, correlation preservation, privacy (DCR), and utility (TSTR/TRTR) scores - alongside the real-vs-synthetic visual comparison.
4. **Export:** Download your privacy-preserved synthetic dataset as a CSV.

> **Note on training time:** For datasets with a large row count, training can take 20–30 minutes to produce a good-quality synthetic dataset. For larger datasets specifically, use a minimum of 100 epochs - CTGAN tends to converge well with fewer epochs on bigger datasets, so 100+ is usually enough rather than needing the higher epoch counts smaller datasets may require. Epochs are capped at 1000 and generated row count at 50,000 given this app's shared, CPU-only hosting.

## Findings

> **⚠️ These numbers need to be re-measured.** The table below was generated before several fixes to the evaluation and generation pipeline: synthetic output used to be blanket-rounded to 2 decimal places regardless of each column's real precision (affecting every KS-test and TSTR/TRTR number below), and the categorical/continuous column detection used a cardinality threshold that misclassified low-cardinality numeric columns on small datasets - Iris, at 150 rows, is exactly the size that bug hit hardest. The qualitative takeaways below are likely still directionally correct, but the specific figures should not be cited until these four experiments are rerun against the current code.

Testing across four datasets of varying size and structure revealed that the right training configuration depends heavily on the dataset itself - there's no single epoch count that works best everywhere:

| Dataset | Rows | Best Epochs Found | TSTR / TRTR Gap | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Titanic | ~890 | ~250 (plateaus early) | ~0.11 | Mostly categorical features; accuracy gap stayed flat from 250 to 750 epochs - additional training gave no benefit. |
| Wine Quality | ~1,100 | 750 | ~0.18 | Strongly epoch-sensitive: gap fell from 0.46 (250 epochs) to 0.18 (750 epochs), then degraded at 1,000 epochs - a clear overfitting point where privacy (DCR) also declined. |
| Adult Income | ~48,000 | 100 | ~0.033 (0.8787 vs 0.8459 via XGBoost) | Strong convergence on complex conditionals (e.g., gender-income interactions), but exhibits utility loss on extreme zero-inflated features (capital-gain). |
| Iris | 150 | -- | Unreliable | Too few rows for a stable held-out test split; TSTR/TRTR accuracy varied by several points between identical reruns. Statistical similarity, correlation, and DCR metrics remained meaningful. |

**Takeaways:**

* Larger, simpler (more categorical) datasets tend to converge quickly and are less sensitive to epoch count.
* Smaller datasets with complex, continuous, correlated features (like wine chemistry) benefit substantially from more training - but only up to a point; excessive training reduced both utility and the synthetic data's distance from real records.
* Increasing CTGAN's network capacity (`generator_dim`/`discriminator_dim`) did **not** help wine quality's accuracy gap at a fixed epoch budget - it was epoch count, not model size, that mattered.
* **Zero-inflation & boundary smearing causes utility leaks:** On Adult Income, the ~3.3% accuracy gap in tree models (XGBoost) was largely driven by `capital-gain`. In real data, 91%+ of records are exactly `0`, giving tree models a sharp split criterion. CTGAN smoothed this into a continuous distribution (only ~3.2% exact zeros in synthetic output) and overshot the hard cap of 99,999 up to 105,500+, reducing split purity on the margins.

## ⚠️ Known Limitations

* **Utility metrics on small datasets:** TSTR/TRTR needs a held-out real test split to compare against, and on datasets under a few hundred rows that split is too small to give a stable reading - accuracy can swing several points between identical runs, as seen on Iris. Statistical similarity, correlation, and DCR don't have this problem and stay reliable regardless of dataset size.
* **No formal privacy guarantee:** DCR is an empirical distance-based privacy signal, not a formal guarantee like differential privacy. It's one useful check, not a certification that the data is anonymous.
* **CTGAN's adversarial training can be unstable:**  The loss curves show real oscillation during training. More epochs isn't always better, which is exactly what the wine quality results above show.
* **Zero-inflated and hard-bounded distributions:** Continuous features with massive point-masses at zero (e.g., `capital-gain`, `capital-loss`) or hard clipping thresholds (e.g., 99,999) can be "smeared" into continuous bell curves by standard mode-specific normalization. While general correlations remain intact, exact boundary-dependent models (like gradient boosted trees) may experience a minor drop in split precision.

## 🚀 Roadmap & Planned Improvements

* [ ] **Zero-Inflation Preprocessor:** Add an automated two-stage transformation for continuous columns with >50% zero-mass (separating binary presence indicators from positive-value log distributions).
* [ ] **Hard Boundary Clamping:** Enforce min/max bounding on synthetic generation to prevent synthetic values from overshooting physical or artificial survey caps.
* [ ] **Expanded Model Suite:** Add XGBoost and LightGBM alongside Random Forest in the automated TSTR/TRTR evaluation benchmark.
