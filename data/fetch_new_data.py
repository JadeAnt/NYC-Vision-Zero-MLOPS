import requests
from datetime import datetime, timedelta
import os
import pandas as pd
import subprocess

end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=7)

url = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"

where = (
    f"crash_date >= '{start_date.isoformat()}' "
    f"AND crash_date < '{(end_date + timedelta(days=1)).isoformat()}'"
)

params = {
    "$where": where,
    "$limit": 50000,          
    "$order": "crash_date DESC"
}

resp = requests.get(url, params=params)
resp.raise_for_status()     


data = resp.json()

print(f"Fetched {len(data)} records from {start_date} through {end_date}")


csv_filename = f"new_data_{end_date.strftime('%Y-%m-%d')}.csv"
df = pd.DataFrame(data)


df.columns = [col.replace('_', ' ').upper() for col in df.columns]

df.to_csv(csv_filename, index=False)
print(f"Saved data to {csv_filename}")


rclone_container = "object-persist-project1"

upload_cmd = [
    "rclone", "copy", csv_filename, 
    f"chi_tacc:{rclone_container}/production",
    "--progress"
]

print(f"Uploading {csv_filename} to object store...")
subprocess.run(upload_cmd, check=True)
print(f"Successfully uploaded {csv_filename} to {rclone_container}")

# Cleanup local file
os.remove(csv_filename)
print(f"Removed local file {csv_filename}")


