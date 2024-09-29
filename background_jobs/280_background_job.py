import subprocess
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format = "{asctime} - {levelname} - {message}",style = "{",datefmt = "%Y-%m-%d %H:%M:%S",)
    logging.info("Starting 280 Background Job...")
    process_data_args = ['--major_road=I280']
    predictions_args = ['--config_filename=config_280.yaml', '--output_filename=data/280_dcrnn_predictions.npz']
    store_predictions_args = ['--major_road=280']

    logging.info("Processing data...")
    subprocess.call(['python', 'process_data.py', ] + process_data_args)
    logging.info("Generating predictions...")
    subprocess.call(['python', 'run_demo_pytorch.py'] + predictions_args)
    logging.info("Storing predictions...")
    subprocess.call(['python', 'store_predictions.py'] + store_predictions_args)
    logging.info(f"280 Background Job Completed")
