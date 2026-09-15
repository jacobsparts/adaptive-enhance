//! The CAIP 2017 exposure fusion framework (Ying et al.).
//!
//! This is the contrast enhancement *after* the illumination estimation, i.e.
//! the part of the paper that the adaptive enhancement does not cover:
//!
//! ```text
//! I     = min/max normalised input, f64 in [0, 1]
//! t     = IlluminationEstimator::estimate(I)         (relative illumination)
//! isBad = t < 0.5                                    (under-exposed pixels)
//! J     = applyK(I, k*) - 0.01                       (camera response model)
//! k*    = argmax_k entropy(geometric mean of applyK(I, k) over isBad)   k in [1, 7]
//! W     = t^mu
//! out   = clamp(I * W + J * (1 - W), 0, 1) * 255     (8-bit result)
//! ```
//!
//! The reference is a PyTorch implementation of the same pipeline. Small
//! numerical details are reproduced deliberately: the entropy
//! is computed on `clip(value * 255, 0, 255) as uint8` histograms, the 50x50
//! optimisation image is produced with `area` resizing of the *normalised* image
//! and the `isBad` mask with a bicubic resize to 50x50 followed by a `>= 0.5`
//! threshold, and the camera response uses `a = -0.3293`, `b = 1.1258`.

use crate::resize::{area_resize_rgb, bicubic_resize_bool};

/// Illumination estimation built on the adaptive enhancement
/// ([`crate::adaptive_enhance_f64`], the port of OpenCE's
/// `adaptiveImageEnhancement()`).
///
/// The exposure fusion framework needs nothing more than a relative
/// illumination map, so the adaptive enhancement's enhanced value channel is
/// normalised and returned as one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveEnhancementEstimator {
    /// Illumination map is normalised before it is returned; the PyTorch
    /// reference normalises `t_our` the same way before building the weights.
    pub normalize: bool,
}

impl Default for AdaptiveEnhancementEstimator {
    fn default() -> Self {
        Self { normalize: true }
    }
}

impl AdaptiveEnhancementEstimator {
    /// The name of this estimator in reports and CLI help.
    pub fn name(&self) -> &'static str {
        "adaptive-enhancement"
    }

    /// Estimate the illumination map of an interleaved `f64` RGB image in
    /// `[0, 1]`, returned as `width * height` values in `[0, 1]`.
    pub fn estimate(&self, rgb: &[f64], width: usize, height: usize) -> Vec<f64> {
        assert_eq!(rgb.len(), width * height * 3, "rgb buffer size mismatch");
        // The adaptive enhancement works on the 0..=255 sample range of the
        // original, so the shared [0, 1] input is scaled as it is converted,
        // which saves a full-image copy of it.
        let (_, value) = crate::adaptive_enhance_interleaved(rgb, width, height, 255.0);
        // The enhanced value channel is still on the 0..=255 scale.
        let mut map: Vec<f64> = value.iter().map(|v| v / 255.0).collect();
        if self.normalize {
            let (min, max) = min_max(&map);
            let range = max - min;
            for v in map.iter_mut() {
                // The adaptive enhancement scales V by (255 + k) / (max(V, Vg) + k),
                // which can exceed 1 for bright pixels; the framework expects a
                // relative illumination in [0, 1].
                *v = ((*v - min) / (range + 1e-15)).clamp(0.0, 1.0);
            }
        }
        map
    }
}

/// Parameters of the camera response model (`applyK` in the PyTorch reference).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraResponse {
    /// Exposure ratio exponent (`a = -0.3293`).
    pub a: f64,
    /// Exposure ratio gain (`b = 1.1258`).
    pub b: f64,
    /// Offset subtracted from the synthesised exposure (`0.01`).
    pub offset: f64,
}

impl Default for CameraResponse {
    fn default() -> Self {
        Self {
            a: -0.3293,
            b: 1.1258,
            offset: 0.01,
        }
    }
}

impl CameraResponse {
    /// `applyK(I, k) = I^(k^a) * exp((1 - k^a) * b)`, one sample at a time.
    pub fn apply(&self, value: f64, k: f64) -> f64 {
        let curve = self.for_ratio(k);
        curve.apply(value)
    }

    /// The two per-image constants of `applyK` for one exposure ratio.
    ///
    /// `k^a` and `exp((1 - k^a) * b)` do not depend on the sample, so a whole
    /// image should compute them once ([`CameraCurve::apply`]) rather than once
    /// per pixel: `powf` and `exp` are the most expensive operations in the
    /// synthesise loop, which is 72 million samples on a 24-megapixel frame.
    pub fn for_ratio(&self, k: f64) -> CameraCurve {
        let gamma = k.powf(self.a);
        CameraCurve {
            gamma,
            beta: ((1.0 - gamma) * self.b).exp(),
        }
    }
}

/// `applyK` with its exposure-ratio-dependent constants precomputed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraCurve {
    gamma: f64,
    beta: f64,
}

impl CameraCurve {
    /// `I^gamma * beta`, the parts of `applyK` that depend on the sample.
    #[inline]
    pub fn apply(&self, value: f64) -> f64 {
        value.powf(self.gamma) * self.beta
    }
}

/// Parameters of the exposure fusion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionParams {
    /// Threshold of the under-exposure mask (`0.5`).
    pub bad_threshold: f64,
    /// Size of the image used for the entropy optimisation (`50`).
    pub optimisation_size: usize,
    /// Lower bound of the exposure ratio search (`1`).
    pub k_min: f64,
    /// Upper bound of the exposure ratio search (`7`).
    pub k_max: f64,
    /// Number of golden-section iterations of the search.
    pub k_iterations: usize,
    /// Camera response model.
    pub camera: CameraResponse,
    /// Where the highlight map stops being the identity and the compress-to-
    /// `CEILING` line takes over. When `None` (the default), an auto-knee is
    /// computed from the diffuse highlight ceiling (98.5th percentile of RGB)
    /// scaled from a baseline of 0.80. When `Some(k)`, `k` is used directly
    /// without adjustment.
    pub knee: Option<f64>,
}

impl Default for FusionParams {
    fn default() -> Self {
        Self {
            bad_threshold: 0.5,
            optimisation_size: 50,
            k_min: 1.0,
            k_max: 7.0,
            k_iterations: 100,
            camera: CameraResponse::default(),
            knee: None,
        }
    }
}

/// Intermediate results of the fusion.
#[derive(Debug, Clone)]
pub struct FusionOutput {
    /// The fused 8-bit RGB image (`width * height * 3` bytes).
    pub rgb: Vec<u8>,
    /// Relative illumination map in `[0, 1]` (`width * height` values).
    pub illumination: Vec<f64>,
    /// Exposure ratio selected by the entropy maximisation.
    pub exposure_ratio: f64,
    /// The effective knee after scaling by the input peak brightness.
    pub effective_knee: f64,
}

/// Run the exposure fusion framework on an 8-bit RGB image.
///
/// The buffers are laid out to keep the peak working set down: the normalised
/// input is overwritten by the synthesised exposure (nothing needs it
/// afterwards), and the original input is reconstructed from the source bytes
/// as the blend consumes it, so one full-image `f64` plane plus a third-size
/// `u8` buffer are alive at once. The returned [`FusionOutput`] carries the
/// fused image, the illumination map and the exposure ratio.
pub fn enhance_rgb_with(
    rgb: &[u8],
    width: usize,
    height: usize,
    params: &FusionParams,
) -> FusionOutput {
    assert_eq!(rgb.len(), width * height * 3, "rgb buffer size mismatch");
    assert!(width > 0 && height > 0, "empty image");

    let (min, max) = rgb.iter().fold((u8::MAX, u8::MIN), |(min, max), &v| {
        (min.min(v), max.max(v))
    });
    let range = (max - min) as f64 + 1e-15;

    // One buffer: normalised input, later overwritten by the synthesized
    // exposure, then consumed pixel by pixel into the 8-bit output.
    let mut input: Vec<f64> = rgb
        .iter()
        .map(|v| (*v as f64 - min as f64) / range)
        .collect();

    let illumination = AdaptiveEnhancementEstimator::default().estimate(&input, width, height);

    let is_bad: Vec<bool> = illumination
        .iter()
        .map(|t| *t < params.bad_threshold)
        .collect();
    let exposure_ratio = optimal_exposure_ratio(&input, width, height, &is_bad, params);

    // Overwrite the normalised input with the synthesised exposure, so only one
    // full-image `f64` buffer stays alive.
    let curve = params.camera.for_ratio(exposure_ratio);
    crate::parallel_fill(&mut input, crate::worker_count(), |value, _| {
        *value = curve.apply(*value) - params.camera.offset;
    });

    let effective_knee = match params.knee {
        Some(k) => k.clamp(0.0, 0.9999),
        None => {
            // Find the diffuse highlight ceiling (98.5th percentile of RGB) to scale
            // the base knee (0.80). This automatically drops the knee on underexposed scenes
            // (protecting overcast skies/midtones) and on specular photobox scenes
            // (protecting pale/metallic subjects from highlight blowout), while leaving normal
            // scenes with true highlights near 0.80.
            let mut hist = [0usize; 256];
            for &v in rgb {
                hist[v as usize] += 1;
            }
            let target = (rgb.len() as f64 * 0.985) as usize;
            let mut accum = 0usize;
            let mut diffuse_ceiling = max;
            for (val, &count) in hist.iter().enumerate() {
                accum += count;
                if accum >= target {
                    diffuse_ceiling = val as u8;
                    break;
                }
            }
            let highlight_scale = diffuse_ceiling as f64 / 255.0;
            (0.80 * highlight_scale * highlight_scale).clamp(0.0, 0.9999)
        }
    };

    // The blend needs the original input as well as the synthesised exposure,
    // and it reads the original from the source bytes through the same
    // normalisation instead of a second full-image copy of them.
    let out = crate::blend::blend(
        &crate::blend::BlendContext {
            input: crate::blend::InputView::Bytes {
                rgb,
                min: min as f64,
                range,
            },
            synthetic: &input,
            illumination: &illumination,
            width,
            height,
        },
        effective_knee,
    );

    FusionOutput {
        rgb: out,
        illumination,
        exposure_ratio,
        effective_knee,
    }
}

/// The exposure ratio `k` that maximises the entropy of the synthesised
/// exposure inside the under-exposed region.
pub fn optimal_exposure_ratio(
    input: &[f64],
    width: usize,
    height: usize,
    is_bad: &[bool],
    params: &FusionParams,
) -> f64 {
    let side = params.optimisation_size;

    // Downscale the image to `side x side` with area interpolation. The
    // interleaved form runs the same accumulation as three per-channel
    // resizes would, without the three full-image channel buffers.
    let downscaled = area_resize_rgb(input, width, height, side, side);
    debug_assert_eq!(downscaled.len(), side * side * 3);

    // Geometric mean of the channels (after torch.clamp(., min=0), with the
    // same min/max renormalisation inside `rgb2gm`).
    let (min, max) = min_max(&downscaled);
    let range = max - min;
    let geometric: Vec<f64> = (0..side * side)
        .map(|i| {
            let r = (downscaled[3 * i] - min) / (range + 1e-15);
            let g = (downscaled[3 * i + 1] - min) / (range + 1e-15);
            let b = (downscaled[3 * i + 2] - min) / (range + 1e-15);
            (r * g * b).abs().cbrt()
        })
        .collect();

    // `isBad` mask resized to `side x side` with bicubic interpolation and
    // thresholded at 0.5, like the reference implementation.
    let mask_small = bicubic_resize_bool(is_bad, width, height, side, side);
    let keep: Vec<f64> = geometric
        .iter()
        .zip(&mask_small)
        .filter(|(_, m)| **m >= 0.5)
        .map(|(y, _)| *y)
        .collect();

    if keep.is_empty() {
        return params.k_min;
    }

    let neg_entropy = |k: f64| -> f64 {
        let curve = params.camera.for_ratio(k);
        let mapped: Vec<f64> = keep.iter().map(|y| curve.apply(*y)).collect();
        -entropy(&mapped)
    };

    golden_section_minimize(neg_entropy, params.k_min, params.k_max, params.k_iterations)
}

/// Shannon entropy (base 2) of the values quantised to 8 bits, like
/// `entropy_np()` in the reference implementation.
pub fn entropy(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut histogram = [0usize; 256];
    for value in values {
        let bin = (*value * 255.0).clamp(0.0, 255.0) as u8;
        histogram[bin as usize] += 1;
    }
    let total = values.len() as f64;
    let mut entropy = 0.0f64;
    for count in histogram {
        if count == 0 {
            continue;
        }
        let p = count as f64 / total;
        entropy -= p * p.log2();
    }
    entropy
}

/// Golden-section search for a minimum on `[lower, upper]`, the same algorithm
/// SciPy runs in `fminbound` (default tolerances give ~80 iterations; the
/// result is reported without the tolerated bracket, like the reference).
pub fn golden_section_minimize<F: Fn(f64) -> f64>(
    f: F,
    lower: f64,
    upper: f64,
    iterations: usize,
) -> f64 {
    const R: f64 = 0.618_033_988_749_894_9;
    let mut left = lower;
    let mut right = upper;
    let mut c = right - R * (right - left);
    let mut d = left + R * (right - left);
    let mut fc = f(c);
    let mut fd = f(d);

    for _ in 0..iterations {
        if fc < fd {
            right = d;
            d = c;
            fd = fc;
            c = right - R * (right - left);
            fc = f(c);
        } else {
            left = c;
            c = d;
            fc = fd;
            d = left + R * (right - left);
            fd = f(d);
        }
    }

    if fc < fd {
        c
    } else {
        d
    }
}

fn min_max(values: &[f64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
    }
    (min, max)
}
