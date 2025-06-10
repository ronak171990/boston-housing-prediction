import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

def load_data():
    housing = fetch_california_housing()
    return housing.data, housing.target

def calculate_average_price(prices):
    return np.mean(prices)

def main():
    data, target = load_data()
    average_price = calculate_average_price(target)
    print(f'Average California housing price: {average_price}')

if __name__ == "__main__":
    main()

# Test cases
def test_load_data():
    data, target = load_data()
    assert data.shape[0] == target.shape[0]
    assert data.shape[1] == 8

def test_calculate_average_price():
    sample_prices = np.array([1, 2, 3, 4, 5])
    average = calculate_average_price(sample_prices)
    assert average == 3

test_load_data()
test_calculate_average_price()