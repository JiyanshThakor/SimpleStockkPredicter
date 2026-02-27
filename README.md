StockMarketPredicter README
===========================

Dataset: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset/data
Files used: stocks/AAPL.csv, stocks/NVDA.csv

Performance
-----------
NVDA Train Accuracy: 86.0%
AAPL Test Accuracy: 85.4% 
Cross-stock drop: 0.6%

Features
--------
11 engineered features:
Open, High, Low, Volume
Lag1 (previous close)
SMA5 (5-day moving average)
Range = (High-Low)/Open
Vol_Change = Volume.pct_change()
Price_Change = (Close-Open)/Open
High_Low_Ratio = High/Low
Volume_SMA (5-day volume average)

Model
-----
RandomForestClassifier(
    n_estimators=20,
    max_depth=5,
    min_samples_split=10,
    random_state=40
)

Usage
-----

Training (main.py):
python main.py
Generates StockPredicter.pkl

Prediction (test.py):
python test.py

Input Format
------------
Exactly 11 columns matching training:
pd.DataFrame({
    'Open':[value], 'High':[value], 'Low':[value], 'Volume':[value],
    'Lag1':[value], 'SMA5':[value], 'Range':[value], 'Vol_Change':[value],
    'Price_Change':[value], 'High_Low_Ratio':[value], 'Volume_SMA':[value]
})

Trading Rules
-------------
Confidence >70%: TRADE
Confidence 55-69%: WAIT  
Confidence <55%: SKIP

Code Structure
--------------
main.py     - Train NVDA, test AAPL, save model
test.py     - Load model, predict new data
StockPredicter.pkl - Deployed model (85.4% accuracy)

Files Required
--------------
stocks/
├── NVDA.csv (train)
└── AAPL.csv (test)

Deployed: February 2026
Status: Production ready, cross-stock validated
