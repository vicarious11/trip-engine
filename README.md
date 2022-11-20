# trip-engine

Purpose: To take input trip data, preprocess and save in MYSQL DB. 

USAGE: 

Pass the environment variables - 

1. MONGO_CLIENT="YOUR MONGO CLIENT"
2. GOOGLE_SNAP_API_KEY="YOUR GOOGLE SNAP API KEY"
3. GOOGLE_SNAP_URL="YOUR GOOGLE SNAP URL"
4. MYSQL_USER="YOUR MYSQL USER"
5. MYSQL_PASSWORD="YOUR MYSQL PASSWORD"
6. MYSQL_HOST="YOUR MYSQL HOST"
7. MYSQLDB_NAME="YOUR MYSQL DBNAME"
8. MYSQL_PORT="YOUR MYSQL PORT"
9. MYSQL_TABLE="YOUR MYSQL TABLE"

Invoke input_trip_info function by passing the trip object. 

Execute the following python commands to run the project.

> pip install -r requirements.txt
> python main.py
