import subprocess
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format = "{asctime} - {levelname} - {message}",style = "{",datefmt = "%Y-%m-%d %H:%M:%S",)
    logging.info("Starting 101 Background Job...")
    process_data_args = ['--major_road=I280']
    training_data_args = ['--major_road=US101', '--sensor_ids_file=data/sensor_graph/device_ids_101.txt']
    predictions_args = ['--config_filename=config_101.yaml', '--output_filename=data/101_dcrnn_predictions.npz']
    store_predictions_args = ['--major_road=101']

    logging.info("Processing data...")
    subprocess.call(['python', 'process_data.py', ] + process_data_args)
    logging.info("Generating test data...")
    subprocess.call(['python', 'generate_training_data_bay.py'] + training_data_args)
    logging.info("Generating predictions...")
    subprocess.call(['python', 'run_demo_pytorch.py'] + predictions_args)
    logging.info("Storing predictions...")
    subprocess.call(['python', 'store_predictions.py'] + store_predictions_args)
    logging.info(f"101 Background Job Completed")
