from django.shortcuts import render
from django.http import HttpResponseRedirect
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import random

# Create your views here.
def homepage(request):
    return render(request, 'houseApp/index.html')

def predict(request):
    final_predicted_price = 0
    if request.method == 'POST':
        locations = ['Downtown', 'Suburbs', 'Rural', 'Coastal']
        conditions = [3, 4, 5, 2, 1]
        crime_rates = [2, 3, 4, 1, 5]
        school_ratings = [4, 3, 5, 2, 1]

        data = []
        for i in range(100):
            location = random.choice(locations)
            living_area = random.randint(1000, 3000)
            num_bedrooms = random.randint(2, 5)
            num_bathrooms = random.randint(1, 4)
            lot_size = random.randint(1000, 10000)
            age = random.randint(5, 50)
            condition = random.choice(conditions)
            interest_rate = random.uniform(3.0, 6.0)
            unemployment_rate = random.uniform(4.0, 8.0)
            gdp_growth = random.uniform(1.0, 3.0)
            median_income = random.uniform(50000, 100000)
            crime_rate = random.choice(crime_rates)
            school_rating = random.choice(school_ratings)
            time_on_market = random.randint(10, 90)
            num_rooms = random.randint(4, 10)
            price = random.uniform(200000, 500000)

            data.append([location, living_area, num_bedrooms, num_bathrooms, lot_size, age, condition, interest_rate, unemployment_rate, gdp_growth, median_income, crime_rate, school_rating, time_on_market, num_rooms, price])

            # Create the DataFrame and save it to a CSV file
        df = pd.DataFrame(data, columns=['location', 'living_area', 'num_bedrooms', 'num_bathrooms', 'lot_size', 'age', 'condition', 'interest_rate', 'unemployment_rate', 'gdp_growth', 'median_income', 'crime_rate', 'school_rating', 'time_on_market', 'num_rooms', 'price'])
        df.to_csv('house_data.csv', index=False)

        # Load the dataset
        data = pd.read_csv('house_data.csv')

        # One-hot encode the 'location' feature
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_encoded = encoder.fit_transform(data[['location']]).toarray()
        X = data[['living_area', 'num_bedrooms', 'num_bathrooms', 'lot_size', 'age', 'condition', 'interest_rate', 'unemployment_rate', 'gdp_growth', 'median_income', 'crime_rate', 'school_rating', 'time_on_market', 'num_rooms']]
        X = pd.concat([X, pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out())], axis=1)
        y = data['price']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        score = model.score(X_test, y_test)
        print(f"Model accuracy: {score:.2%}")

        # Get user input
        location = location
        living_area = float(living_area)
        num_bedrooms = int(num_bedrooms)
        num_bathrooms = int(num_bathrooms)
        lot_size = float(lot_size)
        age = int(age)
        condition = int(condition)
        interest_rate = float(interest_rate)
        unemployment_rate = float(unemployment_rate)
        gdp_growth = float(gdp_growth)
        median_income = float(median_income)
        crime_rate = float(crime_rate)
        school_rating = float(school_rating)
        time_on_market = float(time_on_market)
        num_rooms = int(num_rooms)

        # Encode the new location input
        new_location = [[location]]
        new_location_encoded = encoder.transform(new_location).toarray()
        new_data = [[living_area, num_bedrooms, num_bathrooms, lot_size, age, condition, interest_rate, unemployment_rate, gdp_growth, median_income, crime_rate, school_rating, time_on_market, num_rooms] + list(new_location_encoded[0])]

        # Make the prediction
        predicted_price = model.predict(new_data)

        final_predicted_price = int(predicted_price)

        

    context = {
        'predicted_price':final_predicted_price
    }
    return render(request, 'houseApp/predict.html', context)