import numpy as np
import pandas
from pymongo import MongoClient
import json
import math
from datetime import datetime, timedelta
from IPython.display import display
import pandas as pd
import argparse
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)

def process_data(args):
    # query mongodb database collection for traffic data by major road
    client = MongoClient(uuidRepresentation='pythonLegacy')
    db = client.trafficdata
    collection = db.trafficdata

    # TODO update timestamp to get from previous days based on current timestamp
    trafficdata = collection.find({
        'deviceIdNo': {'$exists': True},
        'MAJOR_ROAD': args.major_road,
        'timestamp': {'$gte': 1728312847}
    }).sort({'timestamp': 1})

    trafficdata_list = list(trafficdata)

    print(f"length of traffic data: {len(trafficdata_list)}")

    # keep track of current speed values
    current_speed_values = []
    created_at_dates = []

    # keep track of device ids and device id locations
    device_id_set = set()
    device_id_locations = []

    # keep track of speed values for each device id
    device_id_values = {}
    device_id_values['Date'] = []

    device_id_location_index = 0

    # go through each traffic data record in ascending order by timestamp
    for traffic in trafficdata_list:
        # get speed value from traffic data JSON
        trafficData = json.loads(traffic['trafficData'])
        current_speed = trafficData['flowSegmentData']['currentSpeed']
        current_speed_values.append(current_speed)

        # convert utc to timestamp
        # created_at_timestamp = int(round(datetime.timestamp(traffic['createdAt'])))
        created_at_timestamp = traffic['createdAt'] - timedelta(hours=7)
        if created_at_timestamp not in created_at_dates:
            created_at_dates.append(created_at_timestamp)

        # keep track of distinct device ids
        device_id_int = int(traffic['deviceIdNo'])

        # keep track of distinct device ids
        if device_id_int not in device_id_set:
            print(f"Found device id {device_id_int}")
            device_id_set.add(device_id_int)
            location_split = traffic['location'].split(',')
            if device_id_int not in device_id_values:
                device_id_values[device_id_int] = []
            # map device id to location
            device_id_locations.append({'index': device_id_location_index, 'sensor_id': device_id_int, 'latitude': location_split[0], 'longitude': location_split[1]})
            device_id_location_index += 1
        # add speed value for device id
        device_id_values.get(device_id_int, []).append(current_speed)

    # update dataframe for speed values
    min_value_length = math.inf
    max_value_length = 0

    for device_id in device_id_values.keys():
        if (device_id == 'Date'):
            continue
        print(f"Key: {device_id}")
        print(f"Length of values: {len(device_id_values.get(device_id))}")
        # find minimum length
        if len(device_id_values.get(device_id)) < min_value_length:
            min_value_length = len(device_id_values.get(device_id))
        # find maximum length
        if len(device_id_values.get(device_id)) > max_value_length:
            max_value_length = len(device_id_values.get(device_id))

    print(f"Min Value Length: {min_value_length}")
    print(f"Max Value Length: {max_value_length}")

    # if min != max, need to remove values to meet min values
    if min_value_length != max_value_length:
        max_min_diff = abs(max_value_length - min_value_length)
        print(f"Need to remove {max_min_diff} values")

        # remove N dates
        created_at_dates = created_at_dates[max_min_diff:]
        device_id_values['Date'] = created_at_dates

        # for each device id values remove N values if not equal to min
        print("Removing values...")
        for device_id in device_id_values.keys():
            current_device_id_values = device_id_values[device_id]
            # update values after removing N dates
            if len(current_device_id_values) != min_value_length:
                print("updating...")
                diff_value = abs(len(current_device_id_values) - min_value_length)
                device_id_values[device_id] = current_device_id_values[diff_value:]

    for device_id in device_id_values.keys():
        print(f"Key: {device_id}")
        print(f"Length of values: {len(device_id_values.get(device_id))}")

    # create matrix of speed values timestamps x sensor ids
    speed_values_by_date = []

    for i in range(min_value_length):
        speed_values_current_date = []
        for device_id in device_id_set:
            speed_values_current_date.append(device_id_values.get(device_id)[i])
        speed_values_by_date.append(speed_values_current_date)
    print(f"device ids: {device_id_set}")
    print(f"speed values by date shape: {np.array(speed_values_by_date).shape}")

    device_id_list = []
    for device_id in device_id_set:
        device_id_list.append(device_id)
    print(len(device_id_list))

    # create dataframe for speeds by ids
    speeds_by_ids_df = pd.DataFrame(data=speed_values_by_date, index=created_at_dates, columns=device_id_list)
    display(speeds_by_ids_df)

    # write dataframe to h5 file
    df_path = f"data/PEMS-BAY/{args.major_road}/data.h5"
    speeds_by_ids_df.to_hdf(df_path, key='speed', mode='w')

    # display collected values
    speeds_by_ids_df.plot(title='Actual Values', figsize=(16, 8), legend=True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--major_road', default='', help='Major road to query data for')
    args = parser.parse_args()
    process_data(args)
