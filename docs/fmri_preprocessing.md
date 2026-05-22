# fMRI preprocessing

The NVD fMRI data were preprocessed with fMRIPrep using Docker. The command below records the preprocessing configuration used for the code release.

```bash
docker run --rm -ti \
  -v /mnt/g/nvd/data:/data:ro \
  -v /mnt/g/nvd/fmriprep_out:/out \
  -v /mnt/g/nvd/fmriprepwkdir:/work \
  -v /mnt/g/license.txt:/opt/freesurfer/license.txt:ro \
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
```

The same command is provided as an editable shell script at:

```text
scripts/00_preprocessing/run_fmriprep_docker.sh
```

## Expected outputs

The downstream analysis scripts in this repository assume fMRIPrep derivatives with file names similar to:

```text
sub-001_ses-101_task-*_space-fsLR_den-91k_bold.dtseries.nii
sub-001_ses-101_task-*_desc-confounds_timeseries.tsv
```

## Notes

- Replace the mounted host paths with local paths before running.
- A valid FreeSurfer license file is required.
- Large fMRIPrep outputs should not be committed to this GitHub repository.
