#!/bin/bash

echo "Ending background jobs..."
kill $(pgrep -f 'background_jobs/237_background_job.py')
kill $(pgrep -f 'background_jobs/280_background_job.py')
kill $(pgrep -f 'background_jobs/680_background_job.py')
kill $(pgrep -f 'background_jobs/880_background_job.py')
kill $(pgrep -f 'background_jobs/85_background_job.py')
kill $(pgrep -f 'background_jobs/87_background_job.py')
kill $(pgrep -f 'background_jobs/101_background_job.py')
echo "Background jobs ended"
