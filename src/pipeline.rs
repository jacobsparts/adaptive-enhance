//! The end-to-end entry points.
//!
//! There is one illumination estimator: the adaptive enhancement this crate
//! implements. [`enhance_rgb`] runs the exposure fusion framework around it,
//! whose blend is the highlight map in [`crate::blend`].

use crate::fusion::{enhance_rgb_with, FusionOutput, FusionParams};

/// Parameters of the whole pipeline.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PipelineParams {
    /// Exposure fusion parameters.
    pub fusion: FusionParams,
}

/// Run the exposure fusion framework.
pub fn enhance_rgb(
    params: &PipelineParams,
    rgb: &[u8],
    width: usize,
    height: usize,
) -> FusionOutput {
    enhance_rgb_with(rgb, width, height, &params.fusion)
}

/// Statistics of an illumination map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub under_exposed_fraction: f64,
}

/// Summarise an illumination map.
pub fn map_stats(map: &[f64]) -> MapStats {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut under = 0usize;
    for value in map {
        min = min.min(*value);
        max = max.max(*value);
        sum += *value;
        if *value < 0.5 {
            under += 1;
        }
    }
    let n = map.len().max(1);
    MapStats {
        min,
        max,
        mean: sum / n as f64,
        under_exposed_fraction: under as f64 / n as f64,
    }
}
