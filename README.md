# Ecommerce-Analysis

## Project Overview
This project involves analyzing customer data from an ecommerce company to help them decide whether to focus on their mobile app experience or their website. Using data analysis and machine learning techniques, we built a linear regression model to identify which factors influence customer spending.

## Dataset
The dataset used in this project contains the following features:
- **Avg. Session Length**: Average session time spent by a customer on the website.
- **Time on App**: Average time spent by a customer on the mobile app.
- **Time on Website**: Average time spent by a customer on the website.
- **Length of Membership**: Duration (in years) of a customer’s membership.
- **Yearly Amount Spent**: Total amount spent by a customer in a year (target variable).

## Goals
1. Perform exploratory data analysis (EDA) to understand the relationships between features.
2. Train a linear regression model to predict yearly customer spending based on provided features.
3. Evaluate the model's performance using metrics like MAE, MSE, and RMSE.

## Project Workflow

### 1. Data Exploration
- Conducted initial exploration using `describe()` and correlation matrices.
- Created visualizations to analyze relationships:
  - Joint plots for `Time on Website` vs. `Yearly Amount Spent` and `Time on App` vs. `Yearly Amount Spent`.
  - Pair plot to identify patterns among features.
  - Regression plot to show the relationship between `Length of Membership` and `Yearly Amount Spent`.

### 2. Data Preparation
- Defined features (**X**) and the target variable (**Y**):
  - Features: `Avg. Session Length`, `Time on App`, `Time on Website`, `Length of Membership`.
  - Target: `Yearly Amount Spent`.
- Split the dataset into training (70%) and testing (30%) sets using `train_test_split`.

### 3. Model Training
- Trained a linear regression model using the training set.
- Extracted coefficients to interpret feature importance.

### 4. Model Evaluation
- Used the following metrics to evaluate the model:
  - **MAE** (Mean Absolute Error): Average absolute difference between actual and predicted values.
  - **MSE** (Mean Squared Error): Average squared difference between actual and predicted values.
  - **RMSE** (Root Mean Squared Error): Square root of the MSE, providing an interpretable error measure.
- Visualized the actual vs. predicted values and analyzed the residuals (errors).

## Key Findings
- **Length of Membership** has the strongest positive correlation with yearly spending.
- Time spent on the **mobile app** has a stronger relationship with spending compared to time spent on the website.
- The linear regression model provides reasonable predictions, with an RMSE value indicating good performance.

## Results
### Model Coefficients:
| Feature               | Coefficient |
|-----------------------|-------------|
| Avg. Session Length  | `25.981550`   |
| Time on App          | `38.590159`   |
| Time on Website      | `0.190405`   |
| Length of Membership | `61.279097`   |

### Evaluation Metrics:
- **MAE**: `7.228148653430811`
- **MSE**: `79.81305165097385`
- **RMSE**: `8.9338150669786`

## Visualizations
The following visualizations were created:
1. Pair plots to show relationships between features.
2. Joint plots for `Time on Website` and `Time on App` vs. `Yearly Amount Spent`.
3. Residual plot to analyze the distribution of errors.
4. Scatter plot showing actual vs. predicted values.

## How to Run the Code
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Ecommerce-Analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Ecommerce-Analysis
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook (`ecommerce_analysis.ipynb`) to see the analysis and results.

## Technologies Used
- **Python**: Programming language used for analysis and modeling.
- **Pandas & NumPy**: Data manipulation and numerical computation.
- **Seaborn & Matplotlib**: Data visualization.
- **Scikit-learn**: Machine learning and evaluation metrics.

## Next Steps
- Perform feature engineering to improve the model.
- Explore advanced algorithms (e.g., Random Forests) for better predictions.
- Investigate user segmentation to provide personalized recommendations.

## Author
Anas Meftah
[Your LinkedIn Profile](https://www.linkedin.com/in/anas-meftah)  
[Your GitHub Profile](https://github.com/anesmeftah)

---
Feel free to reach out if you have questions or suggestions for improvement!

