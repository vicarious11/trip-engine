from pymongo import MongoClient
from datetime import datetime, timedelta
import warnings
warnings.simplefilter(action="ignore", category=UserWarning)
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import json
import googlemaps
import polyline
from collections.abc import MutableMapping
import numpy as np
import mysql.connector 
import random
from dotenv import load_dotenv
import os

load_dotenv()

pd.options.mode.chained_assignment = None

client = MongoClient(
    os.getenv("MONGO_CLIENT", default=None)
)
google_maps_api_key = os.getenv("GOOGLE_SNAP_API_KEY", default=None)
google_snap_url = os.getenv("GOOGLE_SNAP_URL", default=None) + os.getenv("GOOGLE_SNAP_API_KEY", default=None)
gmaps = googlemaps.Client(key=google_maps_api_key)
user = os.getenv("MYSQL_USER", default=None)
password = os.getenv("MYSQL_PASSWORD", default=None)
host = os.getenv("MYSQL_HOST", default=None)
db_name = os.getenv("MYSQLDB_NAME", default=None)
port = os.getenv("MYSQL_PORT", default=None)
conn =  mysql.connector.connect(host=host, user=user, port=port, database=db_name, password=password)


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))


def query_from_mongo(imei, startTime, endTime):
    startTime = datetime.strptime(startTime, "%d-%m-%Y %H:%M")
    endTime = datetime.strptime(endTime, "%d-%m-%Y %H:%M")

    result = client.data["packets"].find(
        {
            "imei": str(imei),
            "ts": {
                "$gte": startTime,
                "$lt": endTime,
            },
        }
    )
    trip_data = []
    for item in result:
        trip_data.append(flatten_dict(item))

    if trip_data:
        return pd.DataFrame(trip_data)
    else:
        pd.DataFrame()


def calculate_distances_coords_local(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    # Convert decimal degrees to Radians:
    R = 6372.8 # this is in miles.  For Earth radius in kilometers use 6372.8 km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))

    return R * c



def get_distance_from_google_maps(start_location, end_location, checkpoints):
    parsed_checkpoints = checkpoints
    parsed_start_location = start_location
    parsed_end_location = end_location
    if not parsed_checkpoints:
        distance = gmaps.distance_matrix(
            (parsed_start_location["lat"], parsed_end_location["lon"]),
            (parsed_end_location["lat"], parsed_end_location["lon"]),
            mode="driving",
        )["rows"][0]["elements"][0]["distance"]["value"]
        return distance / 1000
    else:
        start_lat = parsed_start_location["lat"]
        start_lon = parsed_start_location["lon"]
        distance = 0.0
        for item in parsed_checkpoints:
            distance += gmaps.distance_matrix(
                (start_lat, start_lon),
                (item["lat"], item["lon"]),
                mode="driving",
            )["rows"][0]["elements"][0]["distance"]["value"]
            start_lat = item["lat"]
            start_lon = item["lon"]

        distance += gmaps.distance_matrix(
            (start_lat, start_lon),
            (parsed_end_location["lat"], parsed_end_location["lon"]),
            mode="driving",
        )["rows"][0]["elements"][0]["distance"]["value"]
        return distance / 1000


def ratingdrv(data):
    """
    :param data - expects a dataframe of data of each trips
    """

    # data = data[~(data["gps.speed"] <= 5)]
    data["gps.speed"] = data["gps.speed"].astype(float)
    data["gyro.gy"] = data["gyro.gy"].astype(float)
    data["current"] = data["current"].astype(float)
    speed = data["gps.speed"]
    speed = speed.dropna()
    maxspeed = max(speed)  # max speed
    Averagespeed = speed.mean()  # avg speed

    harsh = speed.diff()
    harsh = harsh.dropna()
    if not harsh.empty:
        harshaccleration = max(
            harsh
        )  # harsh acceleration - max diff in two subsequent speeds
        harshbrake = min(harsh)
    else:
        harsh = 0.0
        harshbrake = 0.0
        harshaccleration = 0.0
    # harsh brake - min diff in two subsequent speeds

    gyroy = data[["gps.speed", "gyro.gy"]]
    gyroy = gyroy.dropna()
    gyroy["gyro.gy"] = gyroy[["gyro.gy"]].diff(periods=1)
    gyroy = gyroy.drop(["gyro.gy"], axis=1)
    harshturn = gyroy[
        (gyroy > 30).any(1)
    ]  # dataframe - Have data where `values > 30` for either `speed` or `diff-of-gyroY`
    turnspeed = harshturn["gps.speed"]  # picking only turn speeds
    if len(turnspeed.index) == 0:
        harshturnspeed = 0
        Avgturnspeed = 0
    else:
        harshturnspeed = round(
            max(turnspeed)
        )  # harsh turn speed - max of turn speeds calculated above
        Avgturnspeed = round(
            turnspeed.mean()
        )  # average turn speed - mean of turn speeds calculated above

    current = data[["current"]]
    current[current < 0] = 0
    currentapplied = (current != 0).any(axis=1)
    new_currentapplied = current.loc[currentapplied]
    if not new_currentapplied.empty:
        maximumcurrent = int(new_currentapplied.dropna().max())  # max current
        avgcurrent = int(new_currentapplied.dropna().mean())  # average current
    else:
        maximumcurrent = 0.0
        avgcurrent = 0.0
    conditions = [
        (maximumcurrent <= 99),
        (maximumcurrent > 100) & (maximumcurrent <= 149),
        (maximumcurrent > 150) & (maximumcurrent <= 200),
        (maximumcurrent > 200),
    ]
    ranges = ["good", "average", "belowavg", "poor"]
    fixrange = np.select(conditions, ranges)

    if fixrange == "good":
        currentproduced = 0.75
    elif fixrange == "average":
        currentproduced = 0.50
    elif fixrange == "belowavg":
        currentproduced = 0.25
    else:
        currentproduced = -0.50

    if avgcurrent > 100:
        currentproduced = currentproduced - 0.25
    else:
        currentproduced = currentproduced + 0.25

    if harshaccleration > 15:
        scoreharshaccleration = 0.50
    else:
        scoreharshaccleration = 1
    if harshbrake > -40:
        scoreharshbrake = 1.5
    else:
        scoreharshbrake = 0.75
    if Averagespeed > 30:
        scoreAveragespeed = 0.75
    else:
        scoreAveragespeed = 1.5
    if maxspeed > 49:
        scoremaxspeed = -3
    else:
        scoremaxspeed = 2
    if harshturnspeed > 27:
        scoreharshturnspeed = 0.75
    else:
        scoreharshturnspeed = 1.5
    if Avgturnspeed > 21:
        scoreAvgturnspeed = 0.75
    else:
        scoreAvgturnspeed = 1.5

    ratings = (
        scoreharshaccleration
        + scoreharshbrake
        + scoreAveragespeed
        + scoremaxspeed
        + scoreharshturnspeed
        + scoreAvgturnspeed
        + currentproduced
    )
    ratings = ratings / 2
    ratings = round(ratings, 2)

    if ratings < 0.49:
        ratings = 0.5

    return ratings



def generate_output(data):
    output = {}
    output["imei"] = data.get("bikeId", None)
    output["vehicleNumber"] = data.get("bikeNumber", None)
    output["startTimestamp"] = datetime.strptime(data.get("start", None), "%d-%m-%Y %H:%M")
    output["endTimestamp"] =  datetime.strptime(data.get("end", None), "%d-%m-%Y %H:%M")
    output["odoAtStart"] = data.get("odoAtStart", None)
    output["odoAtEnd"] = data.get("odoAtEnd", None)
    output["customer"] = data.get("client", None)
    output["vehicleClass"] = data.get("vehicleClass", "LCV")
    output["checkpoints"] = data.get("checkpoints", None)
    output["driver"] = data.get("driver", None)
    output["routeId"] = data.get("routeId", None)
    output["tripId"] = data.get("activityId", None)
    if not output["tripId"]:
        output["tripId"] = f"testId{int(random.randint(1,100000))}" 
    start_location = json.loads(data.get("startLocation", None))
    end_location = json.loads(data.get("endLocation", None))
    checkpoints = json.loads(data.get("checkpoints", None))
    if len(checkpoints) < 3:
        checkpoints = None
    else:
        checkpoints = json.loads(checkpoints)

    output["location"] = json.dumps({"start": start_location, "end": end_location})
    mongo_parsed_start_time = datetime.strptime(data["start"],"%d-%m-%Y %H:%M") - timedelta(hours=5.3)
    mongo_parsed_end_time =  datetime.strptime(data["end"],"%d-%m-%Y %H:%M") - timedelta(hours=5.3)
    trip_df = query_from_mongo(
        output["imei"], datetime.strftime(mongo_parsed_start_time,"%d-%m-%Y %H:%M") , datetime.strftime(mongo_parsed_end_time, "%d-%m-%Y %H:%M")
    )
    if trip_df is not None:
        trip_df = trip_df.sort_values(by=["ts"])
        trip_df.reset_index(inplace=True,drop=True)  
        distance = 0.0
        for index, _ in trip_df.iterrows():
            if index + 1 == len(trip_df):
                break
            distance += calculate_distances_coords_local(
                    float(trip_df["gps.lng"].iloc[index]),
                    float(trip_df["gps.lat"].iloc[index]),
                    float(trip_df["gps.lng"].iloc[index + 1]),
                    float(trip_df["gps.lat"].iloc[index + 1]),
                
            )
        if distance > 1000.0:
            distance = get_distance_from_google_maps(
            start_location,
            end_location,
            checkpoints,
            )
        output["distance"] = str(distance)
        trip_df["gps.speed"] = trip_df["gps.speed"].astype(float)
        output["speedMax"] = str(max(trip_df["gps.speed"]))
        output["speedAverage"] = str(round(trip_df["gps.speed"].mean(), 2))
        output["rating"] = str(ratingdrv(trip_df))
        trip_df["soc"] = trip_df["soc"].astype(float)
        output["SSOC"] = str(trip_df["soc"].iloc[0] / 100)
        output["ESOC"] = str(trip_df["soc"].iloc[-1] / 100)
        trip_df["gps.lat"] = trip_df["gps.lat"].astype(float)
        trip_df["gps.lng"] = trip_df["gps.lng"].astype(float)
        output["mapPathEncoded"] = str(polyline.encode(
            list(zip(trip_df["gps.lat"], trip_df["gps.lng"])), 5
        ))
        output["startVolt"] = str(trip_df["voltage"].iloc[0])
        output["endVolt"] = str(trip_df["voltage"].iloc[-1])
    else:
        distance = get_distance_from_google_maps(
            start_location,
            end_location,
            checkpoints,
        )
        output["distance"] = str(distance)
        output["speedMax"] = None
        output["speedAverage"] = None
        output["rating"] = None
        output["SSOC"] = None
        output["ESOC"] = None
        output["startVolt"] = None
        output["endVolt"] = None
        map_path_encode_list = [(start_location["lat"], end_location["lon"])]
        if checkpoints:
            for item in checkpoints:
                map_path_encode_list.append((item["lat"], item["lon"]))
        map_path_encode_list.append((end_location["lat"], end_location["lon"]))
        output["mapPathEncoded"] = str(polyline.encode(map_path_encode_list, 5))

    output["onTripTime"] = str(round(
        (
            datetime.strptime(data["start"], "%d-%m-%Y %H:%M") 
            - datetime.strptime(data["end"], "%d-%m-%Y %H:%M")
        ).total_seconds()
        / 60.0,
        2,
    ))
    output["roadData"] = json.dumps({"speedBump": {"count": 0, "location": []}})
    output["checkpoints"] = str(checkpoints)
    return output


def form_insert_query(data_dict):
    columns=",".join(data_dict.keys())
    value_placeholders=",".join(["%s"] * len(data_dict))
    query = f'INSERT INTO {os.getenv("MYSQL_TABLE")} ({columns}) VALUES ({value_placeholders})'
    return query
        

def input_trip_info(data):
    """Input data to be written into MariaDB.

    Args:
        data (dict): Input trip data.
    """
    cursor = conn.cursor(buffered=True)
    output_dict = generate_output(data)
    sql = form_insert_query(output_dict)
    cursor.execute(sql, list(output_dict.values()))
    cursor.close()
    conn.commit() 


# df = pd.read_csv("Sample_inputs2_06.09 to 20.09.csv")

# for row, ip in df.iterrows():
#     input_trip_info(ip)
   
# output.append(generate_output(ip))
#bikeId,bikeNumber,startTimestamp,endTimestamp,odoAtStart,odoAtEnd,customer,vehicleClass,checkpoints,driver,routeId,activityId,location,distance,speedMax
# df = pd.DataFrame(output)
# df.to_csv("output.csv")
 # df_gps = trip_df[["gps.lat", "gps.lng"]]
       