from src.analyzer import DataAnalyzer

# We use the Titanic dataset link again
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
engine = DataAnalyzer(URL)

# 2. Run
if engine.load_data():
    print("\n--- Starting Analysis ---")
    results = engine.get_column_types()
    
    # 3. Print Results
    for col, dtype in results.items():
        print(f"{col}: {dtype}")
        
    print("\n✅ System Check Passed.")