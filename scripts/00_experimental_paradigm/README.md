# Experimental paradigm code

This directory contains the Psychtoolbox script used for the naturalistic visual scene task in the NVD project.

## Script

- `nvd_naturalistic_visual_scene_task.m`: presents natural images with interleaved blank trials and fixation-color changes. Numeric keys 1--4 are monitored, and trial-level timing is saved to a BIDS-like `events.tsv` file.

This public-release version only contains the naturalistic visual scene task.

## Example

```matlab
nvd_naturalistic_visual_scene_task('subject','001', ...
    'session','101', ...
    'imageFolder','/path/to/stimuli/val2017', ...
    'outputDir','/path/to/task_logs', ...
    'debug',false, ...
    'syncCheck',true);
```

## Dependencies

- MATLAB
- Psychtoolbox
- A local image directory containing the stimulus images

Use `debug=false` and `syncCheck=true` for scanner acquisition.
