#!/bin/bash

# Install required Python dependencies
pip install requests pandas

# Set up cron job to run every 7 days
(crontab -l 2>/dev/null; echo "0 0 */7 * * cd $(pwd) && python get_new_data.py >> fetch_log.txt 2>&1") | crontab -

echo "Scheduled get_new_data.py to run every 7 days."


