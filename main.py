import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("stocks/NVDA.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df['Target'] = (df['Close'] > df['Close'].shift(1)).astype(int)
df['Lag1'] = df['Close'].shift(1)
df['SMA5'] = df['Close'].rolling(5).mean()
df['Range'] = (df['High'] - df['Low']) / df['Open']
df['Vol_Change'] = df['Volume'].pct_change()
df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
df['High_Low_Ratio'] = df['High'] / df['Low']
df['Volume_SMA'] = df['Volume'].rolling(5).mean()

x_train = df[['Open','High','Low','Volume','Lag1','SMA5','Range','Vol_Change','Price_Change','High_Low_Ratio','Volume_SMA']].dropna()
y_train = df['Target'].loc[x_train.index]

df2 = pd.read_csv("stocks/AAPL.csv")

df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2.sort_values('Date').reset_index(drop=True)

df2['Target'] = (df2['Close'] > df2['Close'].shift(1)).astype(int)
df2['Lag1'] = df2['Close'].shift(1)
df2['SMA5'] = df2['Close'].rolling(5).mean()
df2['Range'] = (df2['High'] - df2['Low']) / df2['Open']
df2['Vol_Change'] = df2['Volume'].pct_change()
df2['Price_Change'] = (df2['Close'] - df2['Open']) / df2['Open']
df2['High_Low_Ratio'] = df2['High'] / df2['Low']
df2['Volume_SMA'] = df2['Volume'].rolling(5).mean()

x_test = df2[['Open','High','Low','Volume','Lag1','SMA5','Range','Vol_Change','Price_Change','High_Low_Ratio','Volume_SMA']].dropna()
y_test = df2['Target'].loc[x_test.index]

model = RandomForestClassifier(n_estimators=40, max_depth=5, min_samples_split=10, random_state=40)
model.fit(x_train, y_train)

prediction = model.predict(x_test)
accuracy = accuracy_score(y_test, prediction)

print(f"Accuracy: {accuracy:.1%}")

plt.scatter(range(len(prediction)), prediction, c='red', label='Prediction')
plt.scatter(range(len(y_test)), y_test, c='blue', label='Actual')
plt.legend()
plt.show()

print(f"NVDA train acc: {accuracy_score(y_train, model.predict(x_train)):.1%}")
print(f"AAPL test acc: {accuracy:.1%}")

joblib.dump(model, "StockPredicter_v2.pkl")