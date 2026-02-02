from joblib import parallel_backend
from src.analyzer import DataAnalyzer
from src.generator import DataGenerator

def main():
    # We use the Titanic dataset link again
    URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    engine = DataAnalyzer(URL)

    # Run
    if engine.load_data():
        print("\n--- Starting Analysis ---")
        results = engine.get_column_types()
        
        # Print Results
        for col, dtype in results.items():
            print(f"{col}: {dtype}")
            
        print("\n✅ System Check Passed.")
        
        print("--- Starting Generator Training ---")
        generator = DataGenerator(engine.df)
        with parallel_backend('threading'):
            generator.train(epochs=20)
        
        print("--- Generating Synthetic Data ---")
        fake_passengers = generator.generate(count=20)
        
        print("\n✅ Generated Passengers:")
        print(fake_passengers[['Name', 'Sex' ,'Age', 'Survived']].head())
        
        print("\n✅ System check Passed.")
        
if __name__ == "__main__":
    main()