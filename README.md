# VR & AR Project
This is the project of CS3327 @ SJTU.

## Doc
All changes are in `src/` folder.

- `denoiser.h`
  - We modify `m_sigmacolor` to adjust the penalty for color difference.
- `denoiser.cpp`
  - In `void Denoiser::Reprojection(const FrameInfo &frameInfo)`, we implement the reprojection function as required after `TODO` sign.
  - In `void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor)`, we implement clamping and temporal accumulation as required after `TODO` sign.
  - There are two versions of `Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo)`, one for ordinary denoising and the other for A Trous Wavelet.
    - The ordinary denoising version is ***ANNOTATED***. Check line 67 to 110 for details.
    - The A Trous Wavelet version is ***UNANNOTATED***. Check the code starting from line 117 for details.
- `main.cpp`
  - We add a timer to measure the time cost of denoising.