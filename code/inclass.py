# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# load our libraries setup up our environment for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression # (pip install scikit-learn)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# %% [markdown]
# ## We are going to work through the Data Science Lifecyle exploring each step
# ### Step 1 Define a question...we can also start with the data but today we are doing the question first


# %% [markdown]
# #### **Question: Which feature(s) of a vehicle most contributes to lower Miles Per Gallon?**
# We are going to use a dataset inside of the seaborn package called "mpg"

# %%
df = sns.load_dataset("mpg") 

# %%
print(df.head()) # shows the first 5 rows of the dataframe
 # shows a summary of the dataframe including data types and non-null counts

# %%
print(df.info()) # What do we notice here?

# %%
df.describe()

# %% [markdown]
# ### Now lets explore the data types

# %% [markdown]
# #### Common Python Data Types
#
# - **int**: Integer type, used to represent whole numbers.  
#     *Example*: `x = 5`
#
# - **float**: Floating point type, used for real numbers (decimals).  
#     *Example*: `y = 3.14`
#
# - **str**: String type, used for text data.  
#     *Example*: `name = "Alice"`
#
# - **bool**: Boolean type, represents `True` or `False` values.  
#     *Example*: `is_valid = True`
#
# - **list**: Ordered, mutable collection of items.  
#     *Example*: `numbers = [1, 2, 3]`
#
# - **tuple**: Ordered, immutable collection of items.  
#     *Example*: `point = (10, 20)`
#
# - **dict**: Dictionary type, stores key-value pairs.  
#     *Example*: `person = {"name": "Bob", "age": 30}`
#
# - **set**: Unordered collection of unique items.  
#     *Example*: `unique_numbers = {1, 2, 3}`
#
# - **NoneType**: Represents the absence of a value.  
#     *Example*: `result = None`

# %%
print(df.dtypes) # check for the data types of each column

# %% [markdown]
# #### Common Pandas Data Types
#
# - **int64**: Integer values (whole numbers).  
#     *Example*: `1, 42, -7`
#
# - **float64**: Floating point numbers (decimals).  
#     *Example*: `3.14, -0.001, 100.0`
#
# - **object**: Typically used for text or mixed types (strings, categorical data, or Python objects).  
#     *Example*: `"sedan", "usa", "ford mustang"`
#
# - **bool**: Boolean values (`True` or `False`).  
#     *Example*: `True, False`
#
# - **datetime64[ns]**: Date and time values with nanosecond precision.  
#     *Example*: `2024-06-01 12:00:00`
#
# - **category**: Categorical data for efficient storage of repeated string values.  
#     *Example*: `["low", "medium", "high"]` as categories
#
# - **timedelta[ns]**: Differences between two datetime values (time durations).  
#     *Example*: `2 days, 5 hours`
#
# These data types help pandas optimize storage and operations on your data.

# %% [markdown]
# ### What data type is Cylinders?

# %%
sns.kdeplot(df['cylinders'], fill=True)
plt.title('Density Plot of Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('Density')
plt.show()

# %%
sns.countplot(x='cylinders', data=df)
plt.title('Count of Cars by Number of Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('Count')
plt.show()

# %%
print(df.dtypes) # Are there other features that might need to be changed?

# %% [markdown]
# ### Changing data types

# %%

df['cylinders'] = df['cylinders'].astype('category')
df['model_year'] = df['model_year'].astype('category')

# %%
print(df.dtypes) # check for the data types of each column after the change

# %%
# What is going to happen if we try to plot a density plot of cylinders again?

sns.kdeplot(df['cylinders'], fill=True) 
plt.title('Density Plot of Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('Density')
plt.show()

# %%
# what about the bar plot?
sns.countplot(x='cylinders', data=df)
plt.title('Count of Cars by Number of Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('Count')
plt.show()

# %%
# what about a histogram of cylinders? 
plt.hist(df['cylinders'])
plt.title('Histogram of Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('Frequency')
plt.show()  

# A bar plot is used to display the frequency of categories (discrete values), while a histogram is used to show the 
# distribution of a continuous variable by grouping values into bins.

# In this case, since 'cylinders' is categorical, a bar plot (like countplot) is more appropriate than a histogram.

# %%
plt.hist(df['mpg'], bins=20, edgecolor='black')
plt.title('Histogram of MPG')
plt.xlabel('Miles Per Gallon (mpg)')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ### What about missing data, did we have any?

# %%
# check for missing values
print(df.isnull().sum())    

# %%
## create boxplot for horsepower
sns.boxplot(x='horsepower', data=df)
plt.title('Boxplot of Horsepower')
plt.xlabel('Horsepower')
plt.show()

# %%
# show the rows with missing horsepower
print(df[df['horsepower'].isnull()])    

# %%

# Check if missing 'horsepower' is associated with other features
missing_hp = df['horsepower'].isnull()
print(df[missing_hp].groupby(['cylinders', 'model_year', 'origin']).size())

# Compare summary statistics for rows with and without missing horsepower
print("Summary for rows with missing horsepower:")
print(df[missing_hp].describe(include='all'))
print("\nSummary for rows without missing horsepower:")
print(df[~missing_hp].describe(include='all'))

# %%
# Show means of each variable for rows with and without missing horsepower
means_missing = df[missing_hp].mean(numeric_only=True)
means_not_missing = df[~missing_hp].mean(numeric_only=True)

comparison_df = pd.DataFrame({
    'Missing Horsepower': means_missing,
    'Not Missing Horsepower': means_not_missing
})

print(comparison_df)

# %% [markdown]
# <details>
#   <summary>Click to show/hide content</summary>
# We really don't see much of a pattern, which is good. This likely means there is no systemic problem with the data collections process. It also means we can likely just delete these rows. 
# </details>

# %%
df = df.dropna()
print(df.info())  # confirm that there are no missing values left

# %% [markdown]
# ### Ok now we have a clean dataset with are data types labelled correctly, now what do we do?

# %% [markdown]
# #### **Correlation** is a statistical measure that describes the strength and direction of a relationship between two variables. In data analysis, correlation helps us understand how changes in one variable are associated with changes in another. A positive correlation means that as one variable increases, the other tends to increase as well. A negative correlation means that as one variable increases, the other tends to decrease. Correlation values range from -1 (perfect negative correlation) to +1 (perfect positive correlation), with 0 indicating no linear relationship.

# %%
# Calculate correlation of all numeric variables with mpg
correlations = df.corr(numeric_only=True)['mpg'].sort_values()

print("Correlation of variables with mpg (lower means more negative correlation):")
print(correlations)



# %%
# Visualize the correlations
# Remove 'mpg' from the correlations before plotting
correlations_no_mpg = correlations.drop('mpg')
plt.figure(figsize=(6, 4))
sns.barplot(x=correlations_no_mpg.values, y=correlations_no_mpg.index, orient='h')
plt.title('Correlation of Features with MPG (excluding mpg itself)')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Feature')
plt.show()


# %%
plt.figure(figsize=(8, 5))
plt.scatter(df['weight'], df['mpg'], c='tab:blue', label='MPG vs Weight')
plt.scatter(df['weight'], df['weight'], c='tab:orange', label='Weight')
plt.xlabel('Weight')
plt.ylabel('Value')
plt.title('Scatter Plot: Weight vs MPG (blue) and Weight (orange)')
plt.legend()
plt.show()

# %% [markdown]
# #### Why doesn't this really work?

# %%
from sklearn.preprocessing import MinMaxScaler

numeric_cols = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

plt.figure(figsize=(8, 5))
plt.scatter(df_normalized['weight'], df_normalized['mpg'], c='tab:blue', label='MPG vs Weight (normalized)')
plt.xlabel('Normalized Weight')
plt.ylabel('Normalized MPG')
plt.title('Scatter Plot: Normalized Weight vs Normalized MPG')
plt.legend()
plt.show()


# %%
plt.figure(figsize=(8, 5))
sns.scatterplot(x='weight', y='mpg', data=df, color='tab:blue', label='MPG vs Weight')
sns.regplot(x='weight', y='mpg', data=df, scatter=False, color='red', label='Trend Line')
plt.xlabel('Weight')
plt.ylabel('Miles Per Gallon (mpg)')
plt.title('Scatter Plot: Weight vs MPG with Trend Line')
plt.legend()
plt.show()

# %%
plt.figure(figsize=(8, 5))
sns.scatterplot(x='horsepower', y='mpg', data=df, color='tab:blue', label='MPG vs Horsepower')
sns.regplot(x='horsepower', y='mpg', data=df, scatter=False, color='red', label='Trend Line')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (mpg)')
plt.title('Scatter Plot: Horsepower vs MPG with Trend Line')
plt.legend()
plt.show()

# %% [markdown]
# #### What about the out categorical feature number of cylinders? Do we think more cylinders worse MPG?

# %%
plt.figure(figsize=(8, 5))
sns.boxplot(x='cylinders', y='mpg', data=df)
plt.title('MPG by Number of Cylinders')
plt.xlabel('Number of Cylinders')
plt.ylabel('Miles Per Gallon (mpg)')
plt.show()

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Select features: all columns except 'name' and 'origin' (object types)
X = df.drop(columns=['name', 'origin', 'mpg'])
# Convert categorical columns to dummy variables
X = pd.get_dummies(X, drop_first=True)
y = df['mpg']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and fit the regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.2f}")
print(f"Test R^2: {r2:.2f}")
coef_table = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': reg.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(coef_table)

# %% [markdown]
# ```markdown
# The regression coefficient for horsepower is approximately -0.03. This means that, holding all other features constant, each additional unit increase in horsepower is associated with a decrease of about 0.03 miles per gallon (mpg) in fuel efficiency.
# ```

# %%
import numpy as np

import plotly.express as px

fig_std = px.scatter_3d(df_normalized, x='weight', y='horsepower', z='mpg',
                        color='mpg', color_continuous_scale='Viridis',
                        title='3D Scatter Plot of Normalized MPG, Weight, and Horsepower')
fig_std.update_layout(scene=dict(
    xaxis_title='Normalized Weight',
    yaxis_title='Normalized Horsepower',
    zaxis_title='Normalized MPG'
))
fig_std.update_layout(width=900, height=700)
fig_std.show()
