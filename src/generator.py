import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

class DataGenerator:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.metadata = None


    def train(self, epochs = 100):
        print("--- 1. Detecting Map (Metadata) ---")
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.df)
        
        print(f"--- 2. Training AI for {epochs} epochs ---")
        self.model = CTGANSynthesizer(
            metadata = self.metadata,
            epochs = epochs,
            verbose = True
        )
        self.model.fit(self.df)
        print("--- Training Complete ---")


    def generate(self, count = 100):
        print(f"--- Generating {count} fake rows")
        synthetic_data = self.model.sample(num_rows = count)
        
        # CLEANING GENERATED FAKE DATA
        # 1. Filling missing ages with mean_age
        mean_age = synthetic_data['Age'].mean()
        synthetic_data['Age'] = synthetic_data['Age'].fillna(mean_age)
        
        # 2. Round the age to nearest whole number
        synthetic_data['Age'] = synthetic_data['Age'].round().astype(int).clip(lower = 1) 
        synthetic_data['Age'] = synthetic_data['Age'].clip(lower = 1) # clip(lower = 1) changes all lowest value say 0 to 1
        
        # 3. Round Fare
        synthetic_data['Fare'] = synthetic_data['Fare'].round(2)
        
        return synthetic_data