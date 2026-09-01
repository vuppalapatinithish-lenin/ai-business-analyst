import sqlite3
import pandas as pd

# Connect to our database
connection = sqlite3.connect("business.db")

# Read the sales table
query = "SELECT * FROM sales"

df = pd.read_sql_query(query, connection)

# Close database connection
connection.close()

# Display the data
print("\n--- SALES DATA ---")
print(df.to_string(index=False))