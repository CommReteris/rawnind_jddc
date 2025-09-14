# Junie's Plan

1. Core libs — documentation and type hints sweep.

[100%] - rawproc.py
[65%] - raw.py
[20%] - rawds.py
[40%] - arbitrary_proc_fun.py: module docstring; clarify functions.
[60%] - rawds_manproc.py: module docstring; class and key method docs.
[100%] - rawtestlib.py: module docstring.
[100%] - __init__.py (libs package): module docstring.
[40%] - abstract_trainer.py: class-level docstrings for ImageToImageNN and ImageToImageNNTraining.

2. Tools — add high-level module docstrings and brief function docs without changing behavior.

[100%] - check_dataset.py: module docstring.
[100%] - crop_datasets.py: module docstring; function docs for create_raw_img_crops, crop_paired_dataset, mtrunner.
[95%] - denoise_image.py: module docstring; function docs for add_arguments, load_image, process_image_base, and clarifications elsewhere.
[100%] - prep_image_dataset.py: function docs for get_args, find_cached_result, fetch_crops_list.
[100%] - add_msssim_score_to_dataset_yaml_descriptor.py: module docstring; function docstring for add_msssim_to_dataset_descriptor.

3. Train scripts — add concise module docstrings describing purpose/config interface.

[100%] - train_dc_bayer2prgb.py
[100%] - train_dc_prgb2prgb.py
[100%] - train_denoiser_bayer2prgb.py
[100%] - train_denoiser_prgb2prgb.py

4. Sanity checks — imports and basic tests.

[0%] - Run embedded unittests where present (e.g., rawproc.py, arbitrary_proc_fun.py).
[0%] - If pytest configured, run a quick subset.

5. Progress tracking.

[72%] - Keep this checklist updated; report after each batch.