chi_tacc]
type = swift
user_id = 8f4b33a596bca520cf966ea20d0b525ed159a108bed352a28095f99c13e422d9
application_credential_id = eaf5fc7a19c54c43ab9072c2696fe172
application_credential_secret = XqBhGQsYfSXhot7eUgpYbmaVCFmStXymTbci2GyzGehj3g83XvPkMizaV7Ass9xjMgtkdqDpOypIdOodEejWhQ
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC#!/bin/sh
set -e


RCLONE_REMOTE_PATH="chi_tacc:object-persist-project1/production"
MOUNT_DIR=${MOUNT_DIR:-/mnt/chi_data} 

# Check if rclone.conf exists
if [ ! -f "${RCLONE_CONFIG}" ]; then
  echo "Error: rclone config file not found at ${RCLONE_CONFIG}" >&2
  exit 1
fi

# Create mount directory if it doesn't exist
mkdir -p "${MOUNT_DIR}"

# Mount the remote
echo "Mounting ${RCLONE_REMOTE_PATH} to ${MOUNT_DIR}..."
rclone mount \
    --config="${RCLONE_CONFIG}" \
    "${RCLONE_REMOTE_PATH}" \
    "${MOUNT_DIR}" \
    --daemon \
    --allow-other \
    --vfs-cache-mode writes \
    --poll-interval 15s \
    --dir-cache-time 15s \
    --timeout 5m \
    --retries 3



if ! mount | grep -q "${MOUNT_DIR}"; then
    echo "Mount verification check: waiting for mount to be ready..."
    sleep 5 
    if ! mount | grep -q "${MOUNT_DIR}"; then
        echo "Error: Failed to mount ${RCLONE_REMOTE_PATH} to ${MOUNT_DIR}. Check rclone logs if available." >&2
      
        exit 1
    fi
fi

echo "Mount successful."

# Execute the CMD passed to the entrypoint
echo "Starting application..."
exec "$@"
