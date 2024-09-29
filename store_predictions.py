import pandas
from IPython.display import display
import pandas as pd
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--major_road', type=str, default='')
    args = parser.parse_args()

    predictions_file_path = f'data/{args.major_road}_dcrnn_predictions.npz'
    selected_sensors_file_path = f'data/sensor_graph/device_ids_{args.major_road}.txt'

    print("Predictions: *****\n")
    data = load(predictions_file_path)
    print(data['prediction'].shape)
    prediction_data = np.array(data['prediction'])

    selected_ids = []
    with open(selected_sensors_file_path) as f:
        content = f.read()
        selected_ids = content.split(',')


    # TODO switch from displaying predictions to storing in database
    predictions_df = pd.DataFrame(data=prediction_data[5], columns=selected_ids)
    predictions_df.plot(title='Predictions One Horizon', figsize=(16, 8), legend=True)
    plt.show()

