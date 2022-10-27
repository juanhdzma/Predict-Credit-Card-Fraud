import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
#print(transactions.info())

# How many fraudulent transactions?
print(sum(transactions.isFraud == 1))

# Summary statistics on amount column
transactions["amount"].describe()

# Create isPayment field
transactions["isPayment"] = 1 if transactions["type"] is ("PAYMENT" or "DEBIT") else 0

# Create isMovement field
transactions["isMovement"] = 1 if transactions["type"] is ("CASH_OUT" or "TRANSFER") else 0

# Create accountDiff field
transactions["accountDiff"] = abs(transactions["oldbalanceOrg"] - transactions["oldbalanceDest"] )

# Create features and label variables
features = [transactions["amount"], transactions["isPayment"], transactions["isMovement"], transactions["accountDiff"]]
label = transactions["isFraud"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(features, label, train_size=0.7, test_size=0.3)

# Normalize the features variables
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Fit the model to the training data
model = LogisticRegression()
model.fit(x_train, y_train)

# Score the model on the training data


# Score the model on the test data


# Print the model coefficients


# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction


# Combine new transactions into a single array


# Normalize the new transactions


# Predict fraud on the new transactions


# Show probabilities on the new transactions