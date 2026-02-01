import pandas as pd
class DataAnalyzer:
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
        self.df = None


    def load_data(self):
        self.df = pd.read_csv(self.dataset_path)
        return True


    def get_column_types(self):
        column_type = {}
        for column_name in self.df.columns:
            dtype = self.df[column_name].dtype
            if dtype == 'object' or dtype == 'O':
                column_type[column_name] = "Categorical"
            elif self.df[column_name].nunique() < 20:
                column_type[column_name] = "Categorical"
            else:
                column_type[column_name] = "Numerical"
        return column_type