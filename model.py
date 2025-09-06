import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_model(data_path):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)

    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day'] = pd.to_datetime(df['timestamp']).dt.day
    df['month'] = pd.to_datetime(df['timestamp']).dt.month

    features = ['latitude', 'longitude', 'temperature', 'humidity', 'wind_speed', 'traffic_volume', 'hour', 'day', 'month']
    target = 'pollution_level'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score (Accuracy): {r2:.2f}")

    return model
