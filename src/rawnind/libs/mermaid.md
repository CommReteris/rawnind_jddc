Error: Parse error on line 5:
... -->|No| E[Scale to [0,1] + WB]    D --
-----------------------^
Expecting 'SQE', 'DOUBLECIRCLEEND', 'PE', '-)', 'STADIUMEND', 'SUBROUTINEEND', 'PIPE', 'CYLINDEREND', 'DIAMOND_STOP', 'TAGEND', 'TRAPEND', 'INVTRAPEND', 'UNICODE_TEXT', 'TEXT', 'TAGSTART', got 'SQS'

```mermaid
graph TD
    A[Load RAW via rawpy] --> B[Extract Mono Bayer + Metadata]
    B --> C{Crop/Force RGGB?}
    C -->|Yes| D[Convert to RGGB Pattern]
    C -->|No| E[Scale to 0,1 + WB]
    D --> E
    E --> F{Demosaic?}
    F -->|Yes| G[Demosaic to camRGB via OpenCV]
    F -->|No| H[Save/Return Mono/RGGB]
    G --> I[Apply WB if needed]
    I --> J[Color Space Transform to Profile e.g., Rec.2020]
    J --> K[Gamma if sRGB]
    K --> L[Export to EXR/TIFF via OIIO/OpenEXR]
    H --> M[Error Checks: Exposure, Patterns]
    subgraph "Proposed Modular Structure"
        P1[RawLoader Class: Handles rawpy loading/cropping]
        P2[Processor Class: WB, Demosaic, Color Transform]
        P3[Exporter Class: HDR File I/O with Profiles]
        P1 --> P2 --> P3
    end
    style A fill:#f9f,stroke:#333
    style L fill:#bbf,stroke:#333
```