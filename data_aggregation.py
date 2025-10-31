import json
import pandas as pd
import pymysql
import sshtunnel
import sys
from sshtunnel import SSHTunnelForwarder
import mysql.connector

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Coins to monitor
coins = ["ETH", "SUI"]

# Establish SSH tunnel and fetch data
def fetch_data(connection):
    aggregated_data = {}

    for coin in coins:
        query = f"SELECT close_time, close, volume FROM ethusdt_transform ORDER BY close_time ASC"
        with connection.cursor() as cursor:
            cursor.execute(query)
            if query.lower().startswith('select'):
                df = pd.DataFrame(cursor.fetchall(), columns=['close_time', 'close', 'volume'])
            else:
                connection.commit()
            
            
        # Add to dictionary
        aggregated_data[coin] = df
        
    
    return aggregated_data

def connect_to_mysql_via_ssh():
    # Create SSH tunnel
    tunnel_config = {
        'ssh_address_or_host': ('your_ssh_ip_address', 22),
        'ssh_username': 'your_ssh_username',
        'remote_bind_address': ('127.0.0.1', 3306)
    }
    
    # Add authentication method (password or key)
    tunnel_config['ssh_private_key'] = 'your_ssh.key'
    
    # Create the tunnel
    tunnel = sshtunnel.SSHTunnelForwarder(**tunnel_config)
    tunnel.start()
    
    # Connect to MySQL through the tunnel
    connection = pymysql.connect(
        host='127.0.0.1',
        port=tunnel.local_bind_port,
        user='your_sql_user',
        passwd='your_sql_password',
        db='your_sql_database'
    )
    
    return tunnel, connection

def execute_query(connection, query):
    """Execute a query and return results."""
    with connection.cursor() as cursor:
        cursor.execute(query)
        if query.lower().startswith('select'):
            return cursor.fetchall()
        else:
            connection.commit()
            return cursor.rowcount

try:
    # Connect to the database via SSH tunnel
    tunnel, connection = connect_to_mysql_via_ssh()
    
    data = fetch_data(connection)
    for coin, df in data.items():
        print(f"\n{coin} Data:")
        print(df.dtypes)
        print(df.head())
    
except Exception as e:
    print(f"Connection error: {e}")

finally:
    # Clean up resources
    if 'connection' in locals() and connection:
        connection.close()
    if 'tunnel' in locals() and tunnel:
        tunnel.close()