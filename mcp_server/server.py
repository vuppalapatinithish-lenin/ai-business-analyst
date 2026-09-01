import sqlite3
from mcp.server.fastmcp import FastMCP


# ==========================================
# CREATE MCP SERVER
# ==========================================

mcp = FastMCP("AI Business Analyst")


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_connection():
    return sqlite3.connect("business.db")


# ==========================================
# TOOL 1: TOTAL REVENUE
# ==========================================

@mcp.tool()
def get_total_revenue() -> str:
    """
    Calculate the total revenue from all sales.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT SUM(revenue)
        FROM sales
    """)

    result = cursor.fetchone()[0]

    connection.close()

    if result is None:
        return "No sales data found."

    return f"Total revenue is ₹{result:,.2f}"


# ==========================================
# TOOL 2: TOP PRODUCT
# ==========================================

@mcp.tool()
def get_top_product() -> str:
    """
    Find the product with the highest total revenue.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            product,
            SUM(revenue) AS total_revenue
        FROM sales
        GROUP BY product
        ORDER BY total_revenue DESC
        LIMIT 1
    """)

    result = cursor.fetchone()

    connection.close()

    if result is None:
        return "No sales data found."

    product = result[0]
    revenue = result[1]

    return (
        f"Top product is {product} "
        f"with revenue of ₹{revenue:,.2f}"
    )


# ==========================================
# TOOL 3: REVENUE BY REGION
# ==========================================

@mcp.tool()
def get_revenue_by_region() -> str:
    """
    Calculate total revenue for each region.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            region,
            SUM(revenue) AS total_revenue
        FROM sales
        GROUP BY region
        ORDER BY total_revenue DESC
    """)

    results = cursor.fetchall()

    connection.close()

    if not results:
        return "No sales data found."

    output = []

    for region, revenue in results:
        output.append(
            f"{region}: ₹{revenue:,.2f}"
        )

    return "\n".join(output)


# ==========================================
# TOOL 4: REVENUE BY MONTH
# ==========================================

@mcp.tool()
def get_revenue_by_month() -> str:
    """
    Calculate total revenue for each month.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            month,
            SUM(revenue) AS total_revenue
        FROM sales
        GROUP BY month
    """)

    results = cursor.fetchall()

    connection.close()

    if not results:
        return "No sales data found."

    output = []

    for month, revenue in results:
        output.append(
            f"{month}: ₹{revenue:,.2f}"
        )

    return "\n".join(output)


# ==========================================
# TOOL 5: PRODUCT SALES DETAILS
# ==========================================

@mcp.tool()
def get_product_sales(product: str) -> str:
    """
    Get sales information for a specific product.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            SUM(quantity),
            SUM(revenue)
        FROM sales
        WHERE LOWER(product) = LOWER(?)
    """, (product,))

    result = cursor.fetchone()

    connection.close()

    if result[0] is None:
        return f"No sales found for {product}."

    quantity = result[0]
    revenue = result[1]

    return (
        f"{product} sold {quantity} units "
        f"and generated ₹{revenue:,.2f} revenue."
    )


# ==========================================
# START MCP SERVER
# ==========================================

if __name__ == "__main__":
    mcp.run()