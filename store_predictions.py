import pandas
from IPython.display import display
import pandas as pd
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import argparse
import datetime
from pymongo import MongoClient
import datetime

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)

DAY_IN_NANOSECONDS = 86400000000000

# function to get H:M:S from nanoseconds
def convert_nanoseconds_to_timestamp(current_nanosecond):
    nanoseconds_in_seconds = current_nanosecond // (10 ** 9)
    return datetime.timedelta(seconds=nanoseconds_in_seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--major_road', type=str, default='')
    parser.add_argument('--datasource', type=str, default='')
    args = parser.parse_args()

    # connect to mongo instance
    client = MongoClient(uuidRepresentation='pythonLegacy')
    db = client.trafficdata
    collection = db.trafficprediction

    # load in predictions, selected sensors
    predictions_file_path = f'data/{args.major_road}_dcrnn_predictions.npz'
    selected_sensors_file_path = f'data/sensor_graph/device_ids_{args.major_road}.txt'

    # load in all datasource, collected values
    datasource_file_path = args.datasource
    datasource = load(datasource_file_path)

    # get timestamps recorded from all datasource
    prediction_timestamps = []
    for value in datasource['y']:
        current_timestamp = convert_nanoseconds_to_timestamp(value[0][0][1] * DAY_IN_NANOSECONDS)
        prediction_timestamps.append(str(current_timestamp))
    print(len(prediction_timestamps))


    print("Predictions: *****\n")
    data = load(predictions_file_path)
    print(data['prediction'].shape)
    prediction_data = np.array(data['prediction'])
    ground_truth_data = np.array(data['truth'])

    selected_ids = []
    with open(selected_sensors_file_path) as f:
        content = f.read()
        selected_ids = content.split(',')


    print(f"prediction data shape: {prediction_data[0].shape}")
    display(prediction_data)
    predictions_df = pd.DataFrame(data=prediction_data[0], columns=selected_ids, index=prediction_timestamps)
    # ax = predictions_df.plot(title='Predictions One Horizon', figsize=(15, 8), legend=True)
    # plt.xlabel('Time')
    # plt.ylabel('MPH')
    # plt.ylim(ymin=0)
    # ax.set_xticks(np.arange(len(predictions_df.index)))
    # ax.set_xticklabels([timestamp for timestamp in predictions_df.index], rotation=90)
    # ax.set_xticks(ax.get_xticks()[::4])


    print("Prediction Values: ")
    display(predictions_df)
    for device_id_no in selected_ids:
        print(f"Device ID: {device_id_no}")
        # get dataframe for specific device id
        current_device_id_no_predictions_df = predictions_df[device_id_no]
        # get timestamps
        current_timestamps = list(current_device_id_no_predictions_df.keys())
        # get values
        current_values = list(current_device_id_no_predictions_df.to_numpy())
        current_values_int = list(map(int, current_values))
        current_values = current_values_int

        traffic_prediction_record = {
            'predictionTimestamps': current_timestamps,
            'speedPredictionValues': current_values,
            'timestamp': int(datetime.datetime.now().timestamp()),
            'updatedAt': datetime.datetime.now(),
            'deviceIdNo': int(device_id_no)
        }
        print(traffic_prediction_record)
        print("Inserting prediction record...")
        try:
            collection.replace_one({'deviceIdNo': int(device_id_no)}, traffic_prediction_record, upsert=True)
            print(f"Inserted prediction record for device id no: {device_id_no}")
        except Exception as e:
            print("Error while inserting")
            print(e)

    client.close()
    display(predictions_df)
    # plt.show()
