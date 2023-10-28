from sklearn.linear_model import LinearRegression
from datetime import datetime
from Gold_Price.models import GoldPriceData  # Import your model here

def load_historical_data():
    data = GoldPriceData.objects.all().values('date', 'price')
    dates = [datetime.strptime(entry['date'], '%Y-%m-%d').date() for entry in data]
    prices = [entry['price'] for entry in data]
    return dates, prices

def train_prediction_model(dates, prices):
    X = [[i] for i in range(len(dates))]  # Using simple indices as features
    y = prices
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_gold_price(input_date):
    dates, prices = load_historical_data()

    # Preprocess input date (e.g., convert it to a numerical feature)
    input_date = datetime.strptime(input_date, '%Y-%m-%d').date()
    input_feature = [[(input_date - dates[0]).days]]

    # Train the prediction model
    model = train_prediction_model(dates, prices)

    # Make a prediction
    predicted_price = model.predict(input_feature)

    return predicted_price[0]  # Return the predicted gold price as a float
