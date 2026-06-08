from joblib import parallel_backend
from src.analyzer import analyze_data
from src.generator import TensorVeilGenerator

def main():
    # We use the Titanic dataset link again
    URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    engine = analyze_data(URL)

    # Run
    if engine.load_data():
        print("\n--- Starting Analysis ---")
        results = analyze_data(engine.df)
        
        # Print Results
        for col, dtype in results.items():
            print(f"{col}: {dtype}")
            
        print("\n✅ System Check Passed.")
        
        print("--- Starting Generator Training ---")
        generator = TensorVeilGenerator(engine.df)
        with parallel_backend('threading'):
            generator.train(epochs=20)
        
        print("--- Generating Synthetic Data ---")
        fake_passengers = generator.generate(count=20)
        
        print("\n✅ Generated Passengers:")
        print(fake_passengers[['Name', 'Sex' ,'Age', 'Survived']].head())
        
        print("\n✅ System check Passed.")
        
if __name__ == "__main__":
    main()