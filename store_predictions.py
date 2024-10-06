import pandas
from IPython.display import display
import pandas as pd
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import argparse
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
        print(f"current timestamp: {current_timestamp}")
        prediction_timestamps.append(str(current_timestamp))
    print(len(prediction_timestamps))
    print(prediction_timestamps)


    print("Predictions: *****\n")
    data = load(predictions_file_path)
    print(data['prediction'].shape)
    prediction_data = np.array(data['prediction'])

    ground_truth_data = np.array(data['truth'])

    selected_ids = []
    with open(selected_sensors_file_path) as f:
        content = f.read()
        selected_ids = content.split(',')


    # TODO switch from displaying predictions to storing in database
    print(f"prediction data shape: {prediction_data[0].shape}")
    predictions_df = pd.DataFrame(data=prediction_data[0], columns=selected_ids, index=prediction_timestamps)
    predictions_df.plot(title='Predictions One Horizon', figsize=(16, 8), legend=True)
    plt.xlabel('Time')
    plt.ylabel('MPH')
    print("Prediction Values: ")
    display(predictions_df)
    plt.show()



    truth_df = pd.DataFrame(data=ground_truth_data[0])
    truth_df.plot(title='Ground Truth One Horizon', figsize=(16, 8), legend=True)
    plt.show()

