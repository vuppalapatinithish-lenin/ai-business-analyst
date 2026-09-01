import sqlite3
import pandas as pd

# Connect to SQLite database
connection = sqlite3.connect("business.db")

# Sample sales data
data = {
    "order_id": [1001, 1002, 1003, 1004, 1005,
                 1006, 1007, 1008, 1009, 1010],

    "product": [
        "Laptop",
        "Smartphone",
        "Headphones",
        "Monitor",
        "Laptop",
        "Keyboard",
        "Smartphone",
        "Monitor",
        "Headphones",
        "Laptop"
    ],

    "region": [
        "South",
        "North",
        "South",
        "East",
        "West",
        "South",
        "North",
        "East",
        "West",
        "South"
    ],

    "quantity": [2, 5, 10, 3, 1, 8, 4, 2, 12, 3],

    "price": [
        60000,
        30000,
        3000,
        15000,
        60000,
        2500,
        30000,
        15000,
        3000,
        60000
    ],

    "month": [
        "June",
        "June",
        "June",
        "June",
        "July",
        "July",
        "July",
        "July",
        "August",
        "August"
    ]
}

# Convert data into DataFrame
df = pd.DataFrame(data)

# Calculate revenue
df["revenue"] = df["quantity"] * df["price"]

# Store data in SQLite
df.to_sql("sales", connection, if_exists="replace", index=False)

connection.close()

print("Database created successfully!")
print("Table 'sales' created successfully!")