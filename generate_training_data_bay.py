# generate training data
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import sys
from IPython.display import display

np.set_printoptions(suppress=True)


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    print(f"num_samples: {num_samples}, num_nodes: {num_nodes}")
    # rotates data
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        # subtract timestamps - timestamps converted to datetime64[D] and then divide each by timedelta64
        # take index datetimestamps subtract with values converted to datetime64[D]
        # (df.index.values - df.index.values.astype("datetime64[D]")) is time difference from start of day to timestamp in nanoseconds
        # for each time difference divide by timedelta64("D"), divide difference in nanoseconds / 1 day in nanoseconds
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        # time ind is ratio in nanoseconds for day
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    # format data into [speed, timestamp/day nanosecond ratio]
    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    print(f"x_offset: {x_offsets}, y_offset: {y_offsets}")
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print(f"num_samples: {num_samples}")
    print(f"min_t: {min_t}, max_t: {max_t}")
    for t in range(min_t, max_t):
        # sequence length x_t is used to predict y_t
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(df, args):
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        # seq length 6
        np.concatenate((np.arange(-5, 1, 1),))
    )
    # Predict the next one hour
    # horizon 6
    y_offsets = np.sort(np.arange(1, 2, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )
    # data is transformed into [speed, datetime within 1 day]

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.1)
    num_train = round(num_samples * 0.8)
    num_val = num_samples - num_test - num_train

    print(f"num_samples: {num_samples}, num_test: {num_test}, num_train: {num_train}, num_val: {num_val}")

    # train
    # 80% of data for training
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    # 10% of data for validation
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    # 10% of data for testing
    x_test, y_test = x[-num_test:], y[-num_test:]

    # get all samples
    x_all, y_all = x[:], y[:]

    for cat in ["train", "val", "test", "all"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(f"data/PEMS-BAY/{args.major_road}", "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_file', default=False, type=str, help='Sensor ids txt file')
    parser.add_argument('--major_road', default=False, type=str, help='Major Road')
    parser.add_argument('--use-pems-data', default=False, type=bool, help='Use PEMS datasource')
    args = parser.parse_args()

    if not os.path.exists(f"data/PEMS-BAY/{args.major_road}"):
        os.makedirs(f"data/PEMS-BAY/{args.major_road}")
    df = None
    if args.use_pems_data:
        df = pd.read_hdf("data/pems-bay.h5")
    else:
        df = pd.read_hdf(f"data/PEMS-BAY/{args.major_road}/data.h5")
    with open(args.sensor_ids_file, "r") as f:
        content = f.read()
        columns = content.split(',')

    columns = map(int, columns)
    df_subset = df[columns]
    print(f"subset shape: {df_subset.shape}")
    print("df_subset: ")
    display(df_subset)

    print("Generating training data")
    generate_train_val_test(df_subset, args)
