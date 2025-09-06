import argparse
from src.model import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Urban Pollution ML Model")
    parser.add_argument('--data', type=str, required=True, help="Path to CSV data file")

    args = parser.parse_args()

    train_model(args.data)
