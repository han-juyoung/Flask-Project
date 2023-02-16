import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocesser:
    def __init__(self, data):
        if isinstance(data, dict):
            data = pd.DataFrame(data, index=[0])
        self.data = data
        self.base_data = pd.read_excel("data/train_synch_drop_final.xlsx").drop(
            columns=["TARGET_GDM"]
        )

    # return the scaled data with dataframe.
    def scale(self):
        scaler = StandardScaler()
        self.data = self.data.reindex(sorted(self.base_data.columns), axis=1)
        scaler.fit(self.base_data)
        return scaler.transform(self.data)
