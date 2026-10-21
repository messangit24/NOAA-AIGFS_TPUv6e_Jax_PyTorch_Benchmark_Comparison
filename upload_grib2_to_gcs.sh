#!/usr/bin/env bash
# ==============================================================================
# NOAA AIGFS 6-Hourly GRIB2 Benchmark Output GCS Uploader
# Uploads the 130 6-hourly GRIB2 files and benchmark report to:
#   gs://mb-noaa-eu/outputs/20261019_181217_aigfs_grib2/
# ==============================================================================

set -euo pipefail

RUN_FOLDER="20261019_181217_aigfs_grib2"
LOCAL_DIR="/tmp/${RUN_FOLDER}"
REPORT_FILE="/tmp/xprof_report_2026092900_20261019_181217.txt"
TARGET_URI="gs://mb-noaa-eu/outputs/${RUN_FOLDER}"

if [[ ! -d "${LOCAL_DIR}" ]]; then
    echo "ERROR: Local staging directory not found: ${LOCAL_DIR}" >&2
    exit 1
fi

if [[ -f "${REPORT_FILE}" ]]; then
    cp -f "${REPORT_FILE}" "${LOCAL_DIR}/"
fi

echo "======================================================================"
echo "   UPLOADING AIGFS 6-HOURLY GRIB2 BENCHMARK OUTPUTS TO GCS"
echo "======================================================================"
echo "Local Source:       ${LOCAL_DIR}/ ($(ls -1 "${LOCAL_DIR}"/*.grib2 | wc -l) GRIB2 files)"
echo "Target Destination: ${TARGET_URI}/"
echo ""

# Ensure active account has write permissions to gs://mb-noaa-eu
if ! gcloud storage cp "${REPORT_FILE}" "${TARGET_URI}/xprof_report_2026092900_20261019_181217.txt" 2>/dev/null; then
    echo "Switching to admin@messan.altostrat.com and authenticating for GCS write access..."
    gcloud auth login admin@messan.altostrat.com
fi

echo "Uploading all 130 GRIB2 files and benchmark report to ${TARGET_URI}/ ..."
gcloud storage cp -r "${LOCAL_DIR}" "gs://mb-noaa-eu/outputs/"

echo ""
echo "======================================================================"
echo "   VERIFYING UPLOADED GCS ARTIFACTS"
echo "======================================================================"
gcloud storage ls "${TARGET_URI}/" | head -n 20
echo "..."
echo "Total objects uploaded: $(gcloud storage ls "${TARGET_URI}/**" | wc -l)"
echo "Portal Link: https://gcs-portal-bewk3ak7hq-ez.a.run.app/?prefix=outputs%2F${RUN_FOLDER}%2F"
