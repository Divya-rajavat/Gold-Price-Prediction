# gold_predictor/views.py
from django.shortcuts import render
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn import metrics

data=pd.read_csv("C:/Users/HP/Downloads/Gold_Price_Prediction/Gold_Price_Prediction/static/gold_price_data.csv",encoding='latin-1')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day

x = data[['Year', 'Month', 'Day']]
y = data['GLD']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

lm = linear_model.LinearRegression()
model = lm.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = round(metrics.mean_squared_error(y_test, y_pred), 2)

def home(request):
    return render(request, 'home.html')

def predict(request):
    global model,mse
    date=request.GET.get('date')
    target_date = datetime.strptime(date, '%Y-%m-%d')
    predval = np.array([[target_date.year, target_date.month, target_date.day]])
    prediction = int(model.predict(predval))
    return render(request, 'predict.html', {'prediction':prediction,'mse_score': mse})

def generate_gold_price_distplot():
    # Create the distribution plot
    global y
    sns.set(style="whitegrid")
    sns.histplot(y, kde=False, color="b")
    plt.xlabel('Gold Price')
    plt.ylabel('Frequency')
    plt.title('Gold Price Distribution')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.read()).decode()

def graph(request):
    Graph1 = generate_gold_price_distplot()
    return render(request, 'graph.html',{'Graph1':Graph1})
