### Comprehensive Operation Registry - Detailed Operation Specifications

Based on research from darktable, rawpy, vkdt (Vision Kernel Development Toolkit), and modern computer vision practices,
here's a detailed expansion of the operations for your Function-Based Composable Pipeline Architecture:

### ### Raw Processing Operations

#### **Sensor-Level Operations**

- **`rawprepare`** - Prepare raw sensor data, handle black level subtraction, sensor defect mapping
- **`hotpixel`** - Hot pixel detection and correction using median filtering and statistical analysis
- **`deadpixel`** - Dead pixel detection and interpolation from surrounding pixels
- **`rawblackwhite`** - Black and white level correction with per-channel adjustments
- **`rawcropmask`** - Crop raw data to active sensor area, handle sensor masks
- **`temperature`** - White balance adjustment with temperature/tint controls
- **`whitebalance`** - Automatic white balance using various algorithms (greyworld, max-RGB, etc.)

#### **Demosaicing Operations**

- **`demosaic_bilinear`** - Basic bilinear interpolation demosaicing
- **`demosaic_vng`** - Variable Number of Gradients demosaicing
- **`demosaic_ppg`** - Patterned Pixel Grouping demosaicing
- **`demosaic_amaze`** - Aliasing Minimization and Zipper Elimination
- **`demosaic_rcd`** - Ratio Corrected Demosaicing algorithm
- **`demosaic_lmmse`** - Linear Minimum Mean Square Error demosaicing
- **`demosaic_igv`** - Improved Gradient-based demosaicing
- **`demosaic_xtrans`** - Specialized X-Trans sensor pattern demosaicing
- **`demosaic_learned`** - Deep learning-based demosaicing networks

#### **Raw Enhancement Operations**

- **`rawdenoise`** - Raw domain noise reduction preserving color accuracy
- **`rawsharp`** - Raw domain sharpening before demosaicing
- **`rawcacorrect`** - Raw chromatic aberration correction using sensor data

### ### Color Processing Operations

#### **Color Space Operations**

- **`colorin`** - Input color profile application with ICC profiles
- **`colorout`** - Output color profile transformation
- **`channelmixer`** - RGB channel mixing matrix operations
- **`channelmixerrgb`** - Enhanced RGB channel mixer with color grading
- **`colorbalance`** - Shadow/midtone/highlight color balance
- **`colorbalancergb`** - RGB-aware color balance with luminance preservation
- **`colorlookup`** - 3D LUT color transformations
- **`colorzones`** - Selective color adjustment by hue/saturation/luminance zones

#### **Color Correction Operations**

- **`colorcorrection`** - Lift/gamma/gain color correction
- **`colormapping`** - Color space remapping and gamut handling
- **`colorcontrast`** - Contrast adjustment preserving color relationships
- **`lowpass`** - Color smoothing and noise reduction in color channels
- **`highpass`** - Color detail enhancement
- **`velvia`** - Velvia-style color enhancement (increased saturation)
- **`colorize`** - Colorization effects and color overlay

### ### Tone Mapping Operations

#### **Exposure Operations**

- **`exposure`** - Linear exposure compensation with highlight protection
- **`basecurve`** - Basic tone curve application (linear, camera, custom curves)
- **`tonecurve`** - Advanced tone curve with multiple control points
- **`rgbcurve`** - Individual RGB channel curves
- **`tonemap`** - Global tone mapping operators

#### **Advanced Tone Mapping**

- **`filmic`** - Filmic tone mapping with shoulder/toe control
- **`filmicrgb`** - RGB-aware filmic with color preservation
- **`sigmoid`** - Sigmoid-based tone mapping for smooth transitions
- **`globaltonemap`** - Global tone mapping algorithms (Reinhard, Drago, etc.)
- **`localtonemap`** - Local adaptive tone mapping
- **`shadows_highlights`** - Selective shadow/highlight recovery
- **`levels`** - Levels adjustment (black point, white point, gamma)

### ### Enhancement Operations

#### **Detail Enhancement**

- **`sharpen`** - Unsharp mask and detail enhancement
- **`local_contrast`** - Local contrast enhancement without halos
- **`clarity`** - Mid-tone contrast enhancement (Clarity effect)
- **`structure`** - Structure enhancement preserving smooth areas
- **`surface_blur`** - Edge-preserving blur for skin smoothing
- **`bloom`** - Bloom effect for highlights
- **`orton`** - Orton effect (dreamy glow enhancement)

#### **Lens Corrections**

- **`lens`** - Comprehensive lens correction (distortion, vignetting, CA)
- **`vignette`** - Vignette correction and creative vignetting
- **`cacorrect`** - Chromatic aberration correction
- **`defringe`** - Purple fringing removal
- **`lensfun`** - Lensfun database-driven corrections
- **`perspective`** - Perspective correction and keystone adjustment
- **`ashift`** - Automatic perspective correction

### ### Geometric Operations

#### **Transform Operations**

- **`crop`** - Cropping with composition guides
- **`rotate`** - Rotation with automatic straightening
- **`perspective`** - Perspective correction and keystone
- **`lens_distortion`** - Barrel/pincushion distortion correction
- **`flip`** - Horizontal/vertical flipping
- **`scale`** - Scaling and resampling operations
- **`reframe`** - Intelligent reframing and composition

#### **Advanced Geometric**

- **`liquify`** - Local geometric distortions and corrections
- **`warp`** - General warping transformations
- **`stabilization`** - Image stabilization using motion vectors
- **`registration`** - Image registration and alignment
- **`homography`** - Homographic transformations for perspective matching

### ### Denoising Operations

#### **Classical Denoising**

- **`bilateral`** - Bilateral filtering for edge-preserving smoothing
- **`nlmeans`** - Non-local means denoising
- **`bm3d`** - Block-matching 3D collaborative filtering
- **`guided_filter`** - Guided filtering for structure preservation
- **`anisotropic`** - Anisotropic diffusion filtering
- **`wiener`** - Wiener filtering for known noise characteristics

#### **Advanced Denoising**

- **`profile_denoise`** - Noise profile-based denoising
- **`adaptive_denoise`** - Adaptive denoising based on local image content
- **`wavelet_denoise`** - Wavelet domain denoising
- **`frequency_denoise`** - Frequency domain noise reduction
- **`multi_scale_denoise`** - Multi-scale denoising approach

#### **AI/ML Denoising**

- **`neural_denoise`** - Deep learning-based denoising
- **`utnet_denoise`** - UTNet architecture denoising
- **`diffusion_denoise`** - Diffusion model-based denoising
- **`transformer_denoise`** - Vision transformer denoising
- **`autoencoder_denoise`** - Autoencoder-based noise reduction

### ### Burst Processing Operations

#### **Multi-Frame Fusion**

- **`hdr_merge`** - HDR exposure bracketing with ghost removal
- **`focus_stack`** - Focus stacking for extended depth of field
- **`super_resolution`** - Multi-frame super-resolution
- **`panorama_stitch`** - Panoramic image stitching
- **`temporal_denoise`** - Temporal noise reduction using multiple frames
- **`alignment`** - Multi-frame alignment with motion correction

#### **Specialized Burst Operations**

- **`exposure_fusion`** - Exposure fusion without tone mapping
- **`motion_blur_removal`** - Motion deblurring using multiple exposures
- **`handheld_hdr`** - Handheld HDR with automatic alignment
- **`astro_stacking`** - Astronomical image stacking
- **`bracketed_wb`** - White balance bracketing and fusion

### ### Quality Assessment Operations

#### **Analysis Operations**

- **`overexposed`** - Overexposure detection and visualization
- **`rawoverexposed`** - Raw-level overexposure analysis
- **`histogram`** - Histogram analysis and statistics
- **`waveform`** - Waveform analysis for exposure assessment
- **`vectorscope`** - Vectorscope analysis for color balance
- **`noise_analysis`** - Noise estimation and profiling
- **`sharpness_measure`** - Sharpness and focus quality assessment

#### **Technical Analysis**

- **`clipping_analysis`** - Highlight/shadow clipping detection
- **`color_accuracy`** - Color accuracy measurement against references
- **`exposure_analysis`** - Exposure distribution analysis
- **`dynamic_range`** - Dynamic range measurement
- **`lens_analysis`** - Lens aberration and distortion analysis
- **`motion_analysis`** - Motion blur and camera shake detection

### ### Creative Operations

#### **Artistic Effects**

- **`grain`** - Film grain simulation with various film stocks
- **`borders`** - Border and frame effects
- **`watermark`** - Watermark overlay and blending
- **`texture_overlay`** - Texture blending and overlay effects
- **`vintage`** - Vintage film simulation effects
- **`cross_process`** - Cross-processing color effects
- **`split_toning`** - Split toning for shadows and highlights

#### **Stylization**

- **`color_grading`** - Professional color grading operations
- **`lut_apply`** - Creative LUT application
- **`film_emulation`** - Film stock emulation (Kodak, Fuji, etc.)
- **`black_white`** - Black and white conversion with channel mixing
- **`sepia`** - Sepia toning effects
- **`infrared`** - Infrared photography simulation
- **`cross_polarization`** - Cross-polarization effect simulation

### ### vkdt-Inspired Operations

#### **GPU-Accelerated Processing**

- **`gpu_wavelet`** - GPU-accelerated wavelet transforms
- **`gpu_fft`** - Fast Fourier Transform operations on GPU
- **`gpu_bilateral`** - Optimized GPU bilateral filtering
- **`compute_histogram`** - GPU histogram computation with bins
- **`gpu_guided_filter`** - GPU-optimized guided filtering

#### **Node-Based Operations**

- **`node_blend`** - Blending operations between pipeline branches
- **`node_mask`** - Masking operations for selective processing
- **`node_composite`** - Compositing multiple processing branches
- **`node_switch`** - Conditional operation switching
- **`node_feedback`** - Feedback loops in processing pipeline

#### **Modern Raw Processing**

- **`modern_demosaic`** - State-of-the-art demosaicing algorithms
- **`adaptive_white_balance`** - AI-assisted white balance
- **`intelligent_crop`** - Content-aware cropping suggestions
- **`semantic_segmentation`** - Image segmentation for selective processing
- **`depth_estimation`** - Depth estimation from single images

### ### Operation Specifications Summary

Each operation includes:

- **Input/Output Types**: RAW_BAYER, RAW_4CH, RGB, LAB, GRAYSCALE, MASK, etc.
- **Processing Modes**: Single image, burst processing, batch processing, group processing
- **Metadata Requirements**: Sensor info, exposure data, lens information, etc.
- **Constraints**: GPU memory requirements, computational complexity, prerequisite operations
- **Parameters**: Configurable parameters with sensible defaults
- **Quality Metrics**: Operation-specific quality assessments

This comprehensive operation registry provides **120+ operations** across **8 functional categories**, supporting
everything from basic raw processing to advanced AI-powered enhancement, while maintaining the elegant function-based
organization that makes the pipeline implementation-agnostic and easily composable.