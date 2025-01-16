# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
customers = pd.read_csv("Ecommerce Customers.csv")

# Display the first few rows of the dataset
print(customers.head())

# Exploratory Data Analysis (EDA)
# Statistical summary of the dataset
print(customers.describe(include="all"))

# Correlation matrix for numeric features
print(customers.corr(numeric_only=True))

# Visualizations to explore relationships
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
sns.pairplot(customers)

# Relationship between Length of Membership and Yearly Amount Spent
sns.set_theme(color_codes=True)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)

# Prepare the data for machine learning
# Define the features (X) and target (Y)
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
Y = customers['Yearly Amount Spent']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# Train a Linear Regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Make predictions on the test set
predictions = lm.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Print the evaluation metrics
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')

# Visualize the predictions
plt.scatter(y_test, predictions)
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (predictions)')
plt.title('Actual vs Predicted Values')
plt.show()

# Residual Analysis
residuals = y_test - predictions
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')
plt.show()

# Display the coefficients of the model
coef_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coef_df)
