import pandas as pd
import numpy as np
import json # Add this at the very top of your file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

class CMAPSSDataPipeline:
    def __init__(self):
        # Initialize an ensemble model to handle complex sensor collinearity
        # self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        
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

        # 🚀 ADD THIS EXACT LINE: The Industry Standard C-MAPSS Fix
        df['RUL'] = df['RUL'].clip(upper=125)
        
        print(f"✅ Successfully processed {len(df)} real engine cycles.")
        return df

    # def train_model(self, df):
    #     """Trains the RUL prediction engine"""
    #     print("🧠 Training Random Forest Regressor on sensor degradation patterns...")
        
    #     # Features and Target
    #     X = df[['core_temperature', 'static_pressure', 'fan_speed']]
    #     y = df['RUL']
        
    #     # Train model
    #     self.model.fit(X, y)
        
    #     # Serialize and save the model for the Streamlit frontend
    #     joblib.dump(self.model, 'turbofan_rul_model.pkl')
    #     print("💾 Model successfully trained and saved as 'turbofan_rul_model.pkl'")

    def train_model(self, df):
        print("🧠 Splitting data for evaluation and training...")
        
        # 1. Define Features and Target
        X = df[['core_temperature', 'static_pressure', 'fan_speed']]
        y = df['RUL']
        
        # 2. Split data: 80% to train, 20% held back to test accuracy
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 3. Train the model
        self.model.fit(X_train, y_train)
        
        # 4. Calculate the real R² Accuracy Score (FIXED: Using y_test)
        accuracy = self.model.score(X_test, y_test)
        acc_percentage = accuracy * 100
        print(f"🎯 Model Validation R² Accuracy: {acc_percentage:.2f}%")
        
        # 5. Save the finalized model file
        joblib.dump(self.model, 'turbofan_rul_model.pkl')
        
        # 6. Save the metrics for the Streamlit Dashboard
        metrics = {"r2_accuracy": f"{acc_percentage:.2f}%"}
        with open('model_metrics.json', 'w') as f:
            json.dump(metrics, f)
            
        print("💾 Model and metrics successfully saved.")

if __name__ == "__main__":
    pipeline = CMAPSSDataPipeline()
    processed_df = pipeline.load_and_preprocess_data()
    pipeline.train_model(processed_df)