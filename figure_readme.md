## generate slices from nii.gz

- Use ``script_ct.py``.

- To customize, write a ``_vis_XX()``, like ``_vis_prostate()``; and then change the save name in ``_vis()``.

- To elimate the white boundary in the saved png image, just change the following code section, e.g.,
```commandline
name = "/home/shiqi/cardiac_*_a2_s*.png"
```
