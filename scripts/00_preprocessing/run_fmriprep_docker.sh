#!/usr/bin/env bash
set -euo pipefail

# Run fMRIPrep for the NVD dataset.
#
# Replace the host paths below with paths on your system before running.
# The command below records the preprocessing configuration used for the
# manuscript code release.

BIDS_ROOT="/mnt/g/nvd/data"
OUTPUT_ROOT="/mnt/g/nvd/fmriprep_out"
WORK_DIR="/mnt/g/nvd/fmriprepwkdir"
FS_LICENSE="/mnt/g/license.txt"

mkdir -p "${OUTPUT_ROOT}" "${WORK_DIR}"

docker run --rm -ti \
  -v "${BIDS_ROOT}:/data:ro" \
  -v "${OUTPUT_ROOT}:/out" \
  -v "${WORK_DIR}:/work" \
  -v "${FS_LICENSE}:/opt/freesurfer/license.txt:ro" \
  nipreps/fmriprep:25.2.2 \
  /data /out participant \
  --fs-license-file /opt/freesurfer/license.txt \
  --submm-recon \
  --write-graph \
  --output-spaces MNI152NLin2009cAsym T1w fsLR:den-32k \
  --cifti-output 91k \
  --work-dir /work \
  --omp-nthreads 8 \
  --nthreads 64
