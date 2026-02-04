from ctgan import CTGAN
from joblib import parallel_backend
import pandas as pd

class TensorVeilGenerator:
    def __init__(self, epochs=50):
        self.epochs = epochs
        self.model = CTGAN(epochs=epochs)
    
    
    def train(self, data, categorical_columns):
        with parallel_backend('threading'):
            self.model.fit(data, categorical_columns)
    
    
    def generate(self, count):
        synthetic_data = self.model.sample(count)
        numeric_cols = synthetic_data.select_dtypes(include=['float']).columns
        for col in numeric_cols:
            synthetic_data[col] = synthetic_data[col].round(2)
        return synthetic_data