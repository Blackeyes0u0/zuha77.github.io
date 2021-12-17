# 비트코인 데이터를 통해 주가 예측
2021 fall / AI-x:Deep-learning final project

# Member
물리학과 신중현 2016039534
E-mail : jhshin1030@naver.com
역할 : all


# I. Proposal

- 최근 암호화폐거래소로 투자자들이 몰리면서 비트코인 가격이 급등락하고 있다. 그리하여 딥러닝 모형
을 이용하여 비트코인의 가격을 예측하고, 투자전략을 통해 비트코인의 수익성이 있는지를 분석하는 것이다. 비선형성
과 장기기억 특성을 보이는 비트코인 가격 예측모형으로는 LSTM을 활용하여 분석하였다. 


# II. Datasets
| date | open | close |high|low  |market cap|
| :------------ | :-----------: | -------------------: | -------------------: | -------------: | -------------: | 
| 2013.04.28 | 135.3 | 134.21 |135.98|132.1 |1.49E+09 | 
| 2013.04.30 | 134.44 | 144.54 |147.49|134|1.6E+09 | 
|  |  |  | || | 
|  |  |  | || | 
|  |  |  | || | 
| 2021.12.01 | 56907.97 |57229.83|59041.69|56553.08 |1.08E+12 | [Table caption, works as a reference][section-mmd-tables-table1]

# III. Methodology 

먼저 비트코인의 데이터는 순차적 데이터이다. 
순차적 데이터란 즉, 데이터의 순서가 중요한 일종의 데이터이다. 예를들어 회사 A의 연간 주식시장 가격이라든지. 이런 종류의 데이터에서는 연도별로 확인해야 하기 때문에 순서와 추세를 찾아야 한다. 연도의 순서를 변경이 불가능한 경우이다.

이러한 순차적 데이터를 활용하는 방법들이 생각해보자. 전통적인 Neural Network에서는 이전에 일어난 사건을 바탕으로 나중에 일어나는 사건을 생각하지 못한다. 따라서 RNN은 스스로를 반복하면서 이전단계의 정보를 지속되도록한다.

그 중에서 LSTM (Long Shor-Term Memory)를 이용할것이다. LSTM 은 RNN의 수정된 아키텍처이며, 긴 의존 기간을 필요로 하는 학습을 수행할 능력을 갖고 있다.
모든 RNN은 neural network 모듈을 반복시키는 체인과 같은 형태를 하고 있다. 기본적인 RNN에서 이렇게 반복되는 모듈은 굉장히 단순한 구조를 가지고 있다. 예를 들어 tanh layer 한 층을 들 수 있다.
- RNN의 구조
![png](pg1.png)

LSTM도 똑같이 체인과 같은 구조를 가지고 있지만, 각 반복 모듈은 다른 구조를 갖고 있다. 단순한 neural network layer 한 층 대신에, 4개의 layer가 특별한 방식으로 서로 정보를 주고 받도록 되어 있다.
- LSTM의 
![png](pg2.png)
![png](pg3.png)


장단기 기억에는 다양한 변형이 있으며 제가 설명한 것은 매우 일반적입니다. 모든 LSTM이 위의 예와 같은 것은 아니며 수학 방정식과 LSTM 셀의 작동에서 약간의 차이점을 찾을 수 있습니다. 차이점은 큰 차이점이 아니며 명확하게 이해하면 쉽게 이해할 수 있습니다.

다음은 위에 정의된 개념을 더 깊이 이해할 수 있도록 기본 LSTM 구조를 그래픽으로 나타낸 것입니다.

![png](pg4.jpg)

이제 LSTM의 핵심 개념을 간단히 요약해 보겠습니다.

1. LSTM은 출력되는 것과 별도의 셀 상태를 유지할 수 있습니다.

2. 정보의 흐름을 제어하기 위해 게이트를 사용합니다.

3. Forget Gate는 불필요한 정보를 제거하는 데 사용됩니다.

4. 현재 입력에서 관련 정보 저장

5. 셀 상태를 선택적으로 업데이트

출력 게이트는 
중단 없는 그라데이션 흐름으로 시간 에 따른 셀 상태 역전파  의 필터링된 버전을 반환합니다.

LSTM의 역전파는 RNN 섹션에서 설명한 방식과 유사하게 작동합니다.

- 
# IV. Evaluation & Analysis
- Graphs, tables, any statistics (if any)
# V. Related Work (e.g., existing studies)
- Tools, libraries, blogs, or any documentation that you have used to do this project.
# VI. Conclusion: Discussion

```python
LSTM을 이용한 Bitcoin 예측
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas_datareader.data as pdr

import datetime
from torch.autograd import variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```



```python
#불러올 파일의 경로를 filename 변수에 저장
#filename = 'C:/Users/문서/BTC-USD.csv'
filename = 'C:/Users/문서/BTC.csv'
#filename = 'C:/Users/jhshi/OneDrive/문서/Python Scripts/BTC.csv'
#filename = 'C:/Users/jhshi/Downloads/BTC'
#filename = 'C:/Users/문서/SBUX.csv'
#filename = '/content/SBUX.csv'
df = pd.read_csv(filename)
```


```python
#pandas read_csv로 불러오기
df = pd.read_csv(filename,index_col ='Date' , parse_dates = True)
df.columns
#df = pd.read_csv(filename, index_col = ‘Date’, parse_dates=True)
```




    Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')




```python
df.head()
df.shape
```




    (366, 6)

```python
X = df.iloc[:, :-1]
y = df.iloc[:, 5:6] 
#print(X) 
#print(y)

# open 시가
# high 고가 
# low 저가
# close 종가
# voulume 거래량
# Adj close 주식의 분할, 배당, 배분 등을 고려해 조정한 종가

# 확실한건 거래량 은 데이터에서 제하는것이 중요하다.

```


```python
y_n = np.array(y)
# #print(y_n.T)
len(y_n)
x_n= list(range(len(y_n)))
# #print(x_n)
import matplotlib.pyplot as plt
plt.plot(x_n,y_n)
plt.show()
# print(len(y_n))
```


    
![png](output_9_0.png)
    



```python
# 학습이 잘되기 위해 데이터 정규화
# StandardScaler 각 특징의 평균을 0, 분산을 1이 되도록 변경
# MinMax Scaler  최대 / 최소 값이 각각 1, 0 이 되도록 변경
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mm = MinMaxScaler()
ss = StandardScaler()


X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y) 
```


```python
#first 200 for training

X_train = X_ss[:300, :]
X_test = X_ss[300:, :]

print(X_train.shape)
print(X_test.shape)
y_train = y_mm[:300, :]
y_test = y_mm[300:, :] 
```

    (300, 5)
    (66, 5)
    


```python
print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape) 
```

    Training Shape (300, 5) (300, 1)
    Testing Shape (66, 5) (66, 1)
    


```python
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable 
```


```python
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

print(X_train_tensors.shape)
y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 
```

    torch.Size([300, 5])
    


```python
#reshaping to rows, timestamps, features

X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))


X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 
```


```python
print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape) 
```

    Training Shape torch.Size([300, 1, 5]) torch.Size([300, 1])
    Testing Shape torch.Size([66, 1, 5]) torch.Size([66, 1])
    


```python
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
```


```python

num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 5 #number of features
hidden_size = 1 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 
```


```python
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) #our lstm class 
```


```python
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 
```


```python
for epoch in range(num_epochs):
  outputs = lstm1.forward(X_train_tensors_final) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
  loss = criterion(outputs, y_train_tensors)
 
  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
```

    Epoch: 0, loss: 0.32506
    Epoch: 100, loss: 0.00618
    Epoch: 200, loss: 0.00559
    Epoch: 300, loss: 0.00549
    Epoch: 400, loss: 0.00537
    Epoch: 500, loss: 0.00519
    Epoch: 600, loss: 0.00508
    Epoch: 700, loss: 0.00502
    Epoch: 800, loss: 0.00494
    Epoch: 900, loss: 0.00487
    


```python
df_X_ss = ss.transform(df.iloc[:, :-1]) #old transformers
df_y_mm = mm.transform(df.iloc[:, -1:]) #old transformers

df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))
#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1])) 
```


```python
train_predict = lstm1(df_X_ss)#forward pass
data_predict = train_predict.data.numpy() #numpy conversion
dataY_plot = df_y_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict) #reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=300, c='purple', linestyle='--') #size of the training set

plt.plot(dataY_plot, label='Actuall Data') #actual plot
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show() 
```


    
![png](output_23_0.png)
    



```python

```

