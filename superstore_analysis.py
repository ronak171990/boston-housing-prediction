import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Superstore_analysis.csv", encoding='latin1')

# Basic info
print(df.info())
print(df.describe())
print(df.head())

# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Add month/year columns for trend analysis
df['Month'] = df['Order Date'].dt.to_period('M')

# Monthly sales trend
monthly_sales = df.groupby('Month')['Sales'].sum()

plt.figure(figsize=(12,6))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import seaborn as sns

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Region', y='Sales', estimator=sum, ci=None, palette='Set2')
plt.title("Total Sales by Region")
plt.xticks(rotation=45)
plt.tight_layout()

# Save chart
plt.savefig("sales_by_region.png")
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Discount', y='Profit', hue='Category', alpha=0.7)
plt.title("Profit vs Discount")
plt.tight_layout()

# Save chart
plt.savefig("profit_vs_discount.png")
plt.show()


