### Modern Operation Registry - Deprecated Operations Removed

Based on darktable's evolution and the move toward scene-referred processing, here's the comprehensive operation
registry with deprecated operations removed:

### ### Raw Processing Operations

#### **Sensor-Level Operations**

- **`rawprepare`** - Prepare raw sensor data, handle black level subtraction, sensor defect mapping
- **`hotpixels`** - Hot pixel detection and correction using median filtering and statistical analysis
- **`temperature`** - White balance adjustment with temperature/tint controls (modern version)
- **`rawdenoise`** - Raw domain noise reduction preserving color accuracy

#### **Modern Demosaicing Operations**

- **`demosaic`** - Advanced demosaicing with multiple algorithms (AMaZE, VNG, etc.)
- **`demosaic_learned`** - Deep learning-based demosaicing networks

### ### Color Processing Operations

#### **Scene-Referred Color Operations**

- **`colorin`** - Input color profile application with ICC profiles
- **`colorout`** - Output color profile transformation
- **`channelmixerrgb`** - Modern RGB channel mixer with color grading (replaces old channelmixer)
- **`colorbalancergb`** - Modern RGB-aware color balance (replaces old colorbalance)
- **`primaries`** - Color primaries adjustment for wide gamut workflows

#### **Modern Color Correction Operations**

- **`colorequal`** - Advanced color equalizer
- **`negadoctor`** - Film negative processing

### ### Modern Tone Mapping Operations

#### **Scene-Referred Tone Mapping**

- **`exposure`** - Linear exposure compensation with highlight protection
- **`filmicrgb`** - Modern filmic tone mapping with RGB processing (replaces old filmic)
- **`sigmoid`** - Modern sigmoid tone mapping for smooth transitions
- **`toneequal`** - Tone equalizer for advanced tone control

#### **Highlight Recovery**

- **`highlights`** - Highlight recovery and reconstruction

### ### Enhancement Operations

#### **Modern Detail Enhancement**

- **`sharpen`** - Modern sharpening algorithms
- **`diffuse`** - Diffusion-based enhancement
- **`blurs`** - Modern blur operations

#### **Lens Corrections**

- **`lens`** - Comprehensive lens correction (distortion, vignetting, CA)
- **`cacorrectrgb`** - Modern chromatic aberration correction (replaces cacorrect)
- **`defringe`** - Purple fringing removal
- **`ashift`** - Perspective correction and keystone adjustment

### ### Geometric Operations

#### **Transform Operations**

- **`crop`** - Cropping with composition guides
- **`flip`** - Horizontal/vertical flipping
- **`rotatepixels`** - Pixel-level rotation
- **`scalepixels`** - Scaling operations
- **`clipping`** - Modern cropping and rotation

#### **Advanced Geometric**

- **`liquify`** - Local geometric distortions and corrections
- **`ashift`** - Automatic perspective correction

### ### Modern Denoising Operations

#### **Profile-Based Denoising**

- **`denoiseprofile`** - Modern noise profile-based denoising
- **`nlmeans`** - Non-local means denoising
- **`rawdenoise`** - Raw domain denoising

#### **Classical Denoising (Still Supported)**

- **`bilateral`** - Bilateral filtering (modern implementation)

### ### Quality Assessment Operations

#### **Analysis Operations**

- **`overexposed`** - Overexposure detection and visualization
- **`rawoverexposed`** - Raw-level overexposure analysis

### ### Creative Operations

#### **Modern Creative Effects**

- **`grain`** - Film grain simulation
- **`borders`** - Border and frame effects
- **`watermark`** - Watermark overlay (if Rsvg2 available)
- **`vignette`** - Vignette effects
- **`bloom`** - Bloom effects for highlights

#### **Color Grading**

- **`colorbalancergb`** - Modern color grading
- **`colorzones`** - Color zones adjustment
- **`lut3d`** - 3D LUT application

### ### GPU-Accelerated Operations (Modern Focus)

#### **High-Performance Processing**

- **`rgblevels`** - RGB levels adjustment
- **`rgbcurve`** - RGB curves
- **`levels`** - Modern levels adjustment

### ### Operations Removed (Deprecated)

The following operations have been **deprecated and removed** from this list:

#### **Deprecated Display-Referred Operations**

- ~~`basecurve`~~ - Replaced by scene-referred workflow with exposure + sigmoid/filmic
- ~~`tonecurve`~~ - Replaced by scene-referred tone mapping
- ~~`colorbalance`~~ - Replaced by `colorbalancergb`
- ~~`channelmixer`~~ - Replaced by `channelmixerrgb`

#### **Deprecated Enhancement Operations**

- ~~`atrous`~~ - Replaced by modern sharpening and diffuse
- ~~`equalizer`~~ - Replaced by `toneequal` and `colorequal`
- ~~`lowpass`~~/~~`highpass`~~ - Replaced by modern `blurs` and `sharpen`
- ~~`soften`~~ - Replaced by modern blur operations
- ~~`clahe`~~ - Replaced by modern tone mapping
- ~~`shadhi`~~ - Replaced by `toneequal`
- ~~`lowlight`~~ - Replaced by modern tone mapping
- ~~`relight`~~ - Replaced by scene-referred workflow

#### **Deprecated Color Operations**

- ~~`colorcorrection`~~ - Replaced by `colorbalancergb`
- ~~`colortransfer`~~ - Replaced by modern color operations
- ~~`colormapping`~~ - Replaced by modern color management
- ~~`colorchecker`~~ - Replaced by modern color operations
- ~~`colorcontrast`~~ - Replaced by `colorequal` and tone mapping
- ~~`vibrance`~~ - Replaced by `colorbalancergb` chroma controls
- ~~`velvia`~~ - Replaced by modern color grading
- ~~`monochrome`~~ - Replaced by `channelmixerrgb`
- ~~`colorize`~~ - Replaced by modern color operations

#### **Deprecated Tone Operations**

- ~~`gamma`~~ - Replaced by scene-referred workflow
- ~~`profile_gamma`~~ - Replaced by proper color management
- ~~`tonemap`~~ - Replaced by modern tone mapping
- ~~`globaltonemap`~~ - Replaced by `filmicrgb` and `sigmoid`
- ~~`zonesystem`~~ - Replaced by `toneequal`

#### **Deprecated Noise Reduction**

- ~~`bilat`~~ - Replaced by modern `bilateral` implementation

#### **Deprecated Creative**

- ~~`splittoning`~~ - Replaced by `colorbalancergb`
- ~~`graduatednd`~~ - Replaced by masking + tone operations

#### **Deprecated/Specialty**

- ~~`invert`~~ - Specialized operation, rarely used
- ~~`dither`~~ - Handled automatically in modern workflows
- ~~`finalscale`~~ - Handled automatically
- ~~`useless`~~ - Test module only
- ~~`colisa`~~ - Legacy operation
- ~~`spots`~~ - Replaced by modern `retouch`
- ~~`overlay`~~ - Replaced by modern blending
- ~~`hazeremoval`~~ - Specialized, replaced by tone mapping
- ~~`censorize`~~ - Rarely used specialty operation
- ~~`enlargecanvas`~~ - Replaced by modern cropping
- ~~`mask_manager`~~ - Internal operation
- ~~`rasterfile`~~ - Internal operation

### ### Key Changes in Modern Darktable

#### **1. Scene-Referred Workflow Focus**

- Modern operations focus on scene-referred processing
- Deprecated display-referred operations like `basecurve` and `tonecurve`
- New workflow: `exposure` → `filmicrgb`/`sigmoid` → `colorbalancergb`

#### **2. RGB-Aware Processing**

- New operations work in RGB space: `colorbalancergb`, `channelmixerrgb`, `filmicrgb`
- Deprecated Lab-based operations

#### **3. Unified Operations**

- Multiple old operations consolidated into fewer, more powerful ones
- Example: `colorbalancergb` replaces `colorbalance`, `vibrance`, `splittoning`

#### **4. Modern Algorithms**

- Advanced demosaicing with multiple algorithms
- Profile-based noise reduction
- GPU-accelerated processing

This cleaned registry contains **~45 modern operations** instead of the original 80+ that included many deprecated
modules. The focus is on the current scene-referred workflow that darktable has moved toward since version 3.0+.

Each remaining operation represents the current best practice and is actively maintained in modern darktable versions.