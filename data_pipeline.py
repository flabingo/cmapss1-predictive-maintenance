import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

class CMAPSSDataPipeline:
    def __init__(self):
        # Initialize an ensemble model to handle complex sensor collinearity
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
    def load_and_preprocess_data(self):
        """Ingests and formats the actual NASA C-MAPSS text files"""
        print("📥 Ingesting REAL NASA C-MAPSS sensor telemetry...")
        
        # 1. Define the 26 columns based on NASA's official dataset documentation
        columns = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                  [f'sensor_{i}' for i in range(1, 22)]
        
        # 2. Read the space-separated text file
        # (Make sure 'train_FD001.txt' is in your project folder)
        df = pd.read_csv('nasa_c-mapss-1/train_FD001.txt', sep='\s+', header=None, names=columns)
        
        # 3. Map real NASA sensors to our UI dashboard variables
        # Sensor 3: Total temperature at HPC outlet (°C equivalent)
        # Sensor 11: Static pressure at HPC outlet (psi equivalent)
        # Sensor 9: Physical core speed (RPM equivalent)
        df = df.rename(columns={
            'sensor_3': 'core_temperature',
            'sensor_11': 'static_pressure',
            'sensor_9': 'fan_speed'
        })
        
        # 4. Calculate the true Remaining Useful Life (RUL) for each engine
        # RUL = (Maximum cycles reached by that specific engine) - (Current cycle)
        max_cycles = df.groupby('engine_id')['cycle'].max()
        df['RUL'] = df.apply(lambda row: max_cycles[row['engine_id']] - row['cycle'], axis=1)
        
        print(f"✅ Successfully processed {len(df)} real engine cycles.")
        return df

    def train_model(self, df):
        """Trains the RUL prediction engine"""
        print("🧠 Training Random Forest Regressor on sensor degradation patterns...")
        
        # Features and Target
        X = df[['core_temperature', 'static_pressure', 'fan_speed']]
        y = df['RUL']
        
        # Train model
        self.model.fit(X, y)
        
        # Serialize and save the model for the Streamlit frontend
        joblib.dump(self.model, 'turbofan_rul_model.pkl')
        print("💾 Model successfully trained and saved as 'turbofan_rul_model.pkl'")

if __name__ == "__main__":
    pipeline = CMAPSSDataPipeline()
    processed_df = pipeline.load_and_preprocess_data()
    pipeline.train_model(processed_df)