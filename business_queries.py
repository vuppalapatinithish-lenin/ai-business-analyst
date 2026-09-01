import sqlite3
import pandas as pd

connection = sqlite3.connect("business.db")

# 1. Total Revenue
query = """
SELECT SUM(revenue) AS total_revenue
FROM sales;
"""

result = pd.read_sql_query(query, connection)

print("\n--- TOTAL REVENUE ---")
print(result)


# 2. Highest Revenue Product
query = """
SELECT product, SUM(revenue) AS total_revenue
FROM sales
GROUP BY product
ORDER BY total_revenue DESC
LIMIT 1;
"""

result = pd.read_sql_query(query, connection)

print("\n--- HIGHEST REVENUE PRODUCT ---")
print(result)


# 3. Revenue by Region
query = """
SELECT region, SUM(revenue) AS total_revenue
FROM sales
GROUP BY region
ORDER BY total_revenue DESC;
"""

result = pd.read_sql_query(query, connection)

print("\n--- REVENUE BY REGION ---")
print(result)


# 4. Revenue by Month
query = """
SELECT month, SUM(revenue) AS total_revenue
FROM sales
GROUP BY month
ORDER BY total_revenue DESC;
"""

result = pd.read_sql_query(query, connection)

print("\n--- REVENUE BY MONTH ---")
print(result)


connection.close()