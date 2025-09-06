# Urban Pollution Heatmap - ML Model

This project trains a machine learning model to predict urban pollution levels (e.g., PM2.5, AQI) based on features like location, weather, and traffic, and outputs predictions for heatmap visualization.

## Features

- Input: Latitude, Longitude, Weather, Traffic, Timestamp
- Output: Predicted Pollution Level
- Model: Random Forest Regressor
- Evaluation: R² Score, RMSE

## Usage

```bash
python run_model.py --data data/sample_pollution_data.csv
