# Company price prediction
Predicting company price
<hr>

### What we predict?  

Stock (also capital stock) is all of the shares into which ownership of a corporation is divided. In American English, the shares are collectively known as "stock". A single share of the stock represents fractional ownership of the corporation in proportion to the total number of shares. This typically entitles the stockholder to that fraction of the company's earnings, proceeds from liquidation of assets (after discharge of all senior claims such as secured and unsecured debt), or voting power, often dividing these up in proportion to the amount of money each stockholder has invested. Not all stock is necessarily equal, as certain classes of stock may be issued for example without voting rights, with enhanced voting rights, or with a certain priority to receive profits or liquidation proceeds before or after other classes of shareholders.  

Stock can be bought and sold privately or on stock exchanges, and such transactions are typically heavily regulated by governments to prevent fraud, protect investors, and benefit the larger economy. The stocks are deposited with the depositories in the electronic format also known as Demat account. As new shares are issued by a company, the ownership and rights of existing shareholders are diluted in return for cash to sustain or grow the business. Companies can also buy back stock, which often lets investors recoup the initial investment plus capital gains from subsequent rises in stock price. Stock options, issued by many companies as part of employee compensation, do not represent ownership, but represent the right to buy ownership at a future time at a specified price. This would represent a windfall to the employees if the option is exercised when the market price is higher than the promised price, since if they immediately sold the stock they would keep the difference (minus taxes).

![image](https://user-images.githubusercontent.com/52078955/122673130-bcc2f800-d1d7-11eb-8f07-45e0d1044657.png)

### WARNING!!!

![image](https://user-images.githubusercontent.com/52078955/122673149-d95f3000-d1d7-11eb-9019-b6f378ade82f.png)

It is not investing advice! DO NOT TRUST THESE PREDICTIONS! It was made only for scientific research and educational purposes.

### What I am using?

I use recurent model(LSTM) with 88.000 params. Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, speech recognition and anomaly detection in network traffic or IDSs (intrusion detection systems).  

A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.  

LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the vanishing gradient problem that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs, hidden Markov models and other sequence learning methods in numerous applications

![image](https://user-images.githubusercontent.com/52078955/122673192-001d6680-d1d8-11eb-8d53-0c89b61a5e17.png)

## Let's code it!
For example we'll predict FaceBook(FB)

### Data loading

```python
import pandas_datareader as web
import datetime as dt

company = 'FB'

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2021, 6, 20)

data = web.DataReader(company, 'yahoo', start, end)
```

![image](https://user-images.githubusercontent.com/52078955/122673277-54c0e180-d1d8-11eb-86ee-753c5a8f7677.png)

### Preprocessing

```python
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data['id'] = range(1, len(data) + 1)
data['Date'] = data.index.values

data.drop('Volume', axis=1, inplace=True)
data.drop('Adj Close', axis=1, inplace=True)

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

scaled_data = pd.DataFrame(scaled_data)

scaled_data['id'] = range(1, len(scaled_data) + 1)

scaled_data = scaled_data.rename(columns={0: 'Scaled_data'})

data = pd.merge(data, pd.DataFrame(scaled_data), on='id', how='left').drop('id', axis=1)
```

```python
def winter(x):
  month = str(x).split('-')[1]
  if month in ['12', '01', '02']:
    return 1
  else:
    return 0

def spring(x):
  month = str(x).split('-')[1]
  if month in ['03', '04', '05']:
    return 1
  else:
    return 0

def summer(x):
  month = str(x).split('-')[1]
  if month in ['06', '07', '08']:
    return 1
  else:
    return 0

def autumn(x):
  month = str(x).split('-')[1]
  if month in ['09', '10', '11']:
    return 1
  else:
    return 0

data['Year_price'] = data['Close'].shift(-365)
data['Winter'] = data['Date'].apply(lambda x: winter(x))
data['Spring'] = data['Date'].apply(lambda x: spring(x))
data['Summer'] = data['Date'].apply(lambda x: summer(x))
data['Autumn'] = data['Date'].apply(lambda x: autumn(x))

data.drop('Date', axis=1, inplace=True)

autoreg_columns = ['Year_price']

for i in range(1, 16):
  data['Price_LD_{}'.format(i)] = data['Close'].shift(-i)
  autoreg_columns.append('Price_LD_{}'.format(i))

data['moving_mean_all_autoreg'] = data[autoreg_columns].mean(axis=1)
data['moving_std_all_autoreg'] = data[autoreg_columns].std(axis=1)
data['moving_var_all_autoreg'] = data[autoreg_columns].var(axis=1)
data['moving_min_all_autoreg'] = data[autoreg_columns].min(axis=1)
data['moving_max_all_autoreg'] = data[autoreg_columns].max(axis=1)

data['target'] = data['Close'].shift(-1)

data.dropna(inplace=True)
```

![image](https://user-images.githubusercontent.com/52078955/122673316-846fe980-d1d8-11eb-8255-991e344e619f.png)

```python
from sklearn.model_selection import train_test_split
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'])

print(f'Train shape: {X_train.shape[0]}')
print(f'Test shape: {X_test.shape[0]}')

X_train = np.expand_dims(X_train, 1)
X_test = np.expand_dims(X_test, 1)
```

### Creating model
```python
import tensorflow as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()
```

```python
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

with strategy.scope():
    model = Sequential(
        [
        layers.Dense(16, activation='relu', input_shape=(X_train.shape[1:])),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='elu'),
        layers.Dense(1)
        ]
    )

    model.compile(
        optimizer='adam', loss='mean_squared_error'
    )

model.summary()
```
![image](https://user-images.githubusercontent.com/52078955/122673394-ec263480-d1d8-11eb-8393-be86c14800b8.png)

### Training

```python
epochs = 50
batch_size = 32

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
```

![image](https://user-images.githubusercontent.com/52078955/122673423-08c26c80-d1d9-11eb-9c52-41c7502634c6.png)

## Results
### SMAPE metric

Symmetric mean absolute percentage error (SMAPE or sMAPE) is an accuracy measure based on percentage (or relative) errors. It is usually defined as follows:

![image](https://user-images.githubusercontent.com/52078955/122673472-44f5cd00-d1d9-11eb-9eda-a89b69cbc2be.png)

where At is the actual value and Ft is the forecast value.

The absolute difference between At and Ft is divided by half the sum of absolute values of the actual value At and the forecast value Ft. The value of this calculation is summed for every fitted point t and divided again by the number of fitted points n.

The earliest reference to similar formula appears to be Armstrong (1985, p. 348) where it is called "adjusted MAPE" and is defined without the absolute values in denominator. It has been later discussed, modified and re-proposed by Flores (1986).


```python
def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

preds = model.predict(X_test).flatten()

smape(y_test, preds)
```
*6.966090493791506*

### Graphs
```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(25,10))
plt.plot(range(len(y_test[:150])), y_test[:150], color='black', label=f'Actual {company} price')
plt.plot(range(len(preds[:150])), preds[:150], color='green', label=f'Actual {company} price')
plt.title(f'{company} share price')
plt.xlabel('Time')
plt.ylabel(f'{company} share price')
plt.legend()
plt.show()
```

#### Facebook
![image](https://user-images.githubusercontent.com/52078955/122673568-a6b63700-d1d9-11eb-905c-63f6180d51c2.png)

#### Apple
![image](https://user-images.githubusercontent.com/52078955/122673845-0f51e380-d1db-11eb-9ac2-2d377f07e5b7.png)

#### Netflix
![image](https://user-images.githubusercontent.com/52078955/122673938-999a4780-d1db-11eb-8946-17628b424547.png)

#### Google
![image](https://user-images.githubusercontent.com/52078955/122673659-28a66000-d1da-11eb-9d26-4703d3ce93c1.png)

#### Amazon
![image](https://user-images.githubusercontent.com/52078955/122673796-ce59cf00-d1da-11eb-978c-7daa9d256fe6.png)
