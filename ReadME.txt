## Typical workflow
1. Motion-correct raw imaging movies.
2. Generate projection images for ROI selection.
3. Draw ROI and extract ROI fluorescence traces (`F`, `ΔF/F`, z-score).
4. Align imaging data with FicTrac behavior and recorded trigger signals.
5. Export trial-wise data and summary plots.


##Usage

Example:

```bash
python main.py /path/to/experiment_root
```

## Inputs
The code was developed for datasets containing imaging movies, experiment metadata, FicTrac output, and synchronization/stimulation files.
- `Image_scan_1_region_0_0.tif`
- `Experiment.xml`
- photodiode / synchronization `.h5` file(s)
- FicTrac `.dat` file(s)
- `stim.txt`
- optional `trigger_data.mat`
- optional existing `roi.npy`

## Outputs
Representative outputs include motion-corrected movies, ROI masks, fluorescence trace tables, and aligned trial-level files for downstream analysis and visualization.


## Notes
The current code reflects the data structure used in our experiments and may require adjustment for other acquisition pipelines.