Stock Direction Predictor (85.4% Accuracy)
Production ready UP/DOWN prediction across stocks using scikit-learn RandomForestClassifier!

[
[
[

📌 Results
Metric	Value
NVDA Training Accuracy	86.0%
AAPL Cross-Stock Test	85.4%
Cross-stock Drop	0.6%
Features	11 engineered
Model	RandomForest (max_depth=5)
📁 Dataset
Kaggle Stock Market Dataset
AAPL.csv & NVDA.csv (daily OHLCV data)

🚀 Quick Usage
Train: python main.py → saves StockPredicter.pkl

Predict:

python
model = joblib.load("StockPredicter.pkl")
proba = model.predict_proba(new_data)[0]
print(f"UP: {proba[1]:.1%}, DOWN: {proba[0]:.1%}")
Input must match exactly:

text
Open, High, Low, Volume, Lag1, SMA5, Range, Vol_Change, 
Price_Change, High_Low_Ratio, Volume_SMA
