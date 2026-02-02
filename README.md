# TensorVeil 🛡️

**Welcome to TensorVeil.**

This is a tool for when you need data that *looks* real, but isn't.

Maybe you can't share your actual user database because of privacy laws (GDPR/HIPAA), or maybe you just don't have enough data to train your model. TensorVeil solves that by taking your original dataset, learning its patterns, and "dreaming up" new, synthetic records that are statistically identical to the real thing.

> **Current Status:** 🧠 The Brain is Live (Sprint 2 Complete). We can now train, generate, and clean data from the command line.

---

## ⚡ Why is this cool?

* **It has "Eyes" (The Analyzer):** You don't need to manually tell it which columns are numbers and which are categories. It figures it out.
* **It has a "Brain" (CTGAN):** It doesn't just shuffle data. It uses a Generative Adversarial Network to understand deep relationships (e.g., "People in Class 1 usually pay higher fares").
* **It cleans up after itself:** No more "Age: -5" or "Fare: 12.33333". The generator automatically rounds numbers and sets logical limits so the output looks human-readable immediately.
* **It works on Windows (Seriously):** I patched a specific conflict between Python 3.13 and Windows multiprocessing, so it won't crash your kernel.

---

## 📂 How it's built

```text
TensorVeil/
├── src/
│   ├── analyzer.py       # The Detective: Figures out your data structure
│   └── generator.py      # The Artist: Learns patterns and draws new data
├── main.py               # The Boss: Ties everything together
├── requirements.txt      # The fuel
└── README.md             # You are here

```
## 🏃‍♂️ Quick Start
1. **Get the Dependencies:**\
    You'll need Python 3.10 or newer.

    ```bash
    pip install -r requirements.txt
    ```
2. **Take it for spin**\
    Right now, ```main.py``` is set up to download the Titanic dataset as a test. It will learn from the passengers and generate 20 completely new ones.

    ```bash
    python main.py
    ```
3. **What you'll see**\
    Watch the terminal. First, it analyzes the columns. Then, you'll see a progress bar (that's the AI learning). Finally, it prints your new synthetic passengers:

    | Name | Sex | Age | Survived |
    | :--- | :--- | :--- | :--- |
    | Chibnall, Mrs. (Edith Martha Bowerman) | female | 31 | 0 |
    | Skoog, Master. Harald | male | 4 | 1 |
4. **🐛 The "Windows Fix" (Read this if you use Python 3.13)**\
    If you are running this on Windows with the latest Python, you might know that ```multiprocessing``` is currently broken for some AI libraries (it throws a ```_posixsubprocess``` error).\
    **I fixed it.** In ```main.py```, I forced the system to use **Threading** instead of Processes.

    ```python
    # The magic fix for Windows users
    with parallel_backend('threading'):
        generator.train()
    ```
    *You don't need to do anything extra—it just works.*

## 🗺️ What's Next?
I am currently building this out in sprints.
*   ✅ Sprint 1: Build the Analyzer (Done).
*   ✅ Sprint 2: Build the Generator and fix Windows Bugs (Done).
*   [ ] Sprint 3: Build a UI (Coming soon). Goodbye terminal, hello Drag-and-Drop website!
*   [ ] Sprint 4: Export options & Fine-tuning.