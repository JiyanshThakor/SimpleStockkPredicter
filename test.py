import joblib
import pandas as pd  

model = joblib.load("StockPredicter.pkl")
model2 = joblib.load("StockPredicter_v2.pkl")

new_data = pd.DataFrame({
    'Open':[25000], 'High':[26000], 'Low':[24900], 'Volume':[2000000000],
    'Lag1':[25200], 'SMA5':[25500], 'Range':[0.044], 'Vol_Change':[0.30],
    'Price_Change':[0.04], 'High_Low_Ratio':[1.044], 'Volume_SMA':[1500000000]
})

proba = (model.predict_proba(new_data) + model2.predict_proba(new_data)) / 2
print(f"UP prob: {proba[0,1]:.1%}, DOWN prob: {proba[0,0]:.1%}")