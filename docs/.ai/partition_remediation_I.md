Prompt

- Be sure to use the tools available to you as described in @/.roo/rules/rules.md
- Under no circumstances should you simplify or mock away real parts of the pipeline for integration testing. 

## History/Context
- The main goal of the previous task was to replace synthetic data with real data and ensure the pipeline can execute end-to-end with real dependencies.
- The issue is that the ImageToImageNN classes are trying to parse command line arguments when initialized. You can see in the error that it's looking for required arguments like --arch, --match_gain, --loss. 
- We took a step back and looked at the big picture - you had uncovered an issue in the execution of the refactoring i.e., exactly why we were trying to run this test. 

## Task
- Have a look @/docs/.ai/partition_plan.md , especifically focus on lines 177-208. It is clear that the problematic code was not completely rewritten to the refactoring spec, and instead was just moved from where it lived to reside in the inference package. 
- You will correct that, utilizing PyCharm's advanced python refactoring tools. - The end result should minimize package interdependencies and will make use of clean interfaces for cross package communication (including to the tests you were just working on) - these clean interfaces should completely replace the legacy CLI interfaces
- You should prefer to remove the legacy CLI whenever, rather than deprecate in place, whenever practicable in order to keep a clean codebase. Use PyCharm's advanced search tools to quickly determine whether something is safe to remove (or if it can easily be made safe to remove)
- You will use strict TDD, and begin by inferring the intent of the test_e2e_image_processing_pipeline_real.py integration test, and rewriting it to use your to-be-implemented clean interpackage interfaces. You will also incude tests (for later use - you do not need to do more than write these) that verify the CLI interface no longer exists.
- Focus on one package at a time, beginning with the inference package. You have already begun work on ImageToImageNN.

## Notes:
- You may have to hunt down where missing functions have been incorrectly moved to, and potentially examine previous commits to find code if it is completely missing. 

### Key Technical Concepts:
- E2E Image Processing Pipeline: Complete workflow from RAW/EXR image loading through model inference to output processing
- RawNIND Dataset: Academic dataset from UCLouvain Dataverse containing Bayer RAW and ground truth images
- Bayer vs RGB Processing: 4-channel Bayer pattern images requiring demosaicing vs 3-channel RGB images
- Model Factory Pattern: Denoiser, BayerDenoiser, DenoiseCompress classes inheriting from ImageToImageNN
- PyTorch Model Loading: Using args.yaml configuration files with state_dict loading for trained models
- Test Fixtures and Markers: Real vs synthetic data fixtures with proper pytest markers for test selection
- Image Quality Metrics: MS-SSIM, MSE computation using pt_losses module with proper class instantiation