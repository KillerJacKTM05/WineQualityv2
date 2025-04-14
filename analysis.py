import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.feature_selection import f_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("winequality-red.csv", delimiter=';')
print(df.head())

# Histogram of wine quality
sns.histplot(df['quality'], bins=range(df['quality'].min(), df['quality'].max() + 2), kde=False)
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality Score")
plt.ylabel("Frequency")
plt.show()

print("Quality score range:", df['quality'].min(), "to", df['quality'].max())
# it is 3 to 8.

def quality_label(q):
    if q <= 4:
        return 'low'
    elif q <= 6:
        return 'medium'
    else:
        return 'high'

# Apply, and drop original column
df['quality bin'] = df['quality'].apply(quality_label)
df.drop(columns=['quality'], inplace=True)
print(df['quality bin'].value_counts())

# Pairplot with hue based on quality bin
sns.pairplot(df, hue='quality bin', diag_kind='hist', palette={'low': 'red', 'medium': 'blue', 'high': 'green'})
plt.suptitle("Scatterplot Matrix of Wine Attributes", y=1.02)
plt.show()

# Separate features and target
X = df.drop(columns=['quality bin'])
y = df['quality bin']

# Convert labels to numerical for ANOVA
y_numeric = y.map({'low': 0, 'medium': 1, 'high': 2})

# Apply ANOVA F-test
f_scores, p_values = f_classif(X, y_numeric)

# Wrap into a DataFrame for clarity
anova_df = pd.DataFrame({
    'Feature': X.columns,
    'F-Score': f_scores,
    'p-Value': p_values
}).sort_values(by='F-Score', ascending=False)

# Display top features
print(anova_df)

# Top 5 based on ANOVA
anova_top5 = anova_df['Feature'].head(5).tolist()

# Create new filtered DataFrame
plot_df = df[anova_top5 + ['quality bin']]
print(plot_df.head())

# Initialize PairGrid
g = sns.PairGrid(data=plot_df, hue='quality bin', corner=False)

# Lower triangle: regression plots
g.map_lower(sns.regplot, scatter_kws={'alpha': 0.5}, line_kws={'linewidth': 1.5})
# Upper triangle: scatter plots
g.map_upper(sns.scatterplot, alpha=0.5)
# Diagonal: KDE plots
g.map_diag(sns.kdeplot, fill=True, alpha=0.5)
g.add_legend()
g.fig.suptitle("PairGrid of Top 5 Features", y=1.02)
plt.show()

def distance_consistency(X, y):
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    
    # Check whether nearest neighbor belongs to same class
    return np.mean([y[i] == y[indices[i][1]] for i in range(len(y))])

# Extract top features and quality bin
features = anova_top5
X = df[features]
y = df['quality bin'].values

results = []
for f1, f2 in combinations(features, 2):
    coords = X[[f1, f2]].values
    score = distance_consistency(coords, y)
    results.append((f1, f2, score))

# Sort by score descending
results_df = pd.DataFrame(results, columns=["Feature 1", "Feature 2", "Distance Consistency"])
results_df = results_df.sort_values(by="Distance Consistency", ascending=False)

print(results_df)

# We'll now treat this as a regression problem
# Predict original quality (numeric) instead of bins

# Load dataset again with original quality
df_reg = pd.read_csv("winequality-red.csv", delimiter=';')

# Select top 5 features based on ANOVA analysis
X = df_reg[anova_top5]  # using the same top 5 features
y = df_reg['quality']   # this time, keep quality as a numeric value

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Linear Regression Results ---")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R^2) Score: {r2:.3f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Predicted vs Actual Wine Quality by Linear Regression")
plt.grid(True)
plt.show()

# Train randforest regression model
rand_model = RandomForestRegressor(random_state=42)
rand_model.fit(X_train, y_train)
y_pred = rand_model.predict(X_test)

print("\n--- RandomForest Regression Results ---")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Predicted vs Actual Wine Quality by Random Forest")
plt.grid(True)
plt.show()

