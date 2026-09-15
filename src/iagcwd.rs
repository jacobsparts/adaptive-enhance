//! Improved adaptive gamma correction with weighting distribution.
//!
//! A standalone Rust implementation of the contrast enhancement in
//!
//! > Cao, Gang, et al. "Contrast enhancement of brightness-distorted images by
//! > improved adaptive gamma correction." Computers & Electrical Engineering
//! > 66 (2018): 569-582.
//!
//! It complements the exposure fusion this crate is built around: the fusion
//! lifts underexposed scenes, the gamma correction pulls down over-bright ones.
//! The reference `IAGCWD()` (`image_enhancement.h`) works like this:
//!
//! ```text
//! L = V of BGR -> HSV_FULL             (or the grey image itself)
//! t = (mean(L) - T_t) / T_t
//! t < -tau_t   dimmed : alpha = alpha_dimmed, truncated = false
//! t >  tau_t   bright : alpha = alpha_bright, truncated = true, L = 255 - L
//! otherwise           : copy the input through
//!
//! PDF   = histogram(L) / pixels        (L is inverted before the histogram)
//! PDF_w = pdf_max * ((PDF - pdf_min) / (pdf_max - pdf_min)) ^ alpha
//! CDF_w = cumsum(PDF_w) / sum(PDF_w)
//! gamma = 1 - CDF_w                    (max(tau, 1 - CDF_w) when truncated)
//! table[i] = saturate_cast<uchar>(255 * (i / 255) ^ gamma[i])   table[0] = 0
//! L = LUT(L, table)
//! ```
//!
//! The histogram on the bright path is taken of the plane *after* the
//! inversion, so the weighting distribution of a bright image starts at the
//! dark end of its inverted plane rather than at the top of its value plane.
//!
//! The fixed-point colour conversions, the histogram and
//! `cv::saturate_cast<uchar>` follow OpenCV, which the reference is built on.
//! Four constants settle what the formula above leaves open, each documented
//! where it is applied: [`DECISION_BAND`] widens the "leave the image alone"
//! band around the thresholds, [`MIN_PROB`] bounds the weighting distribution
//! so that a degenerate sample range still yields a finite increasing curve,
//! [`TAB_LOW`] pins the dark end of the table to zero, and an empty histogram
//! bin takes the gamma of the level below it.
//!
//! On a colour image only the value channel changes: hue and saturation are
//! round-tripped unchanged. A single-channel image is corrected directly and
//! stays single-channel.

// ---------------------------------------------------------------------------
// RGB <-> HSV_FULL (8 bit), ported from OpenCV
// ---------------------------------------------------------------------------
//
// Only the value channel is of interest here, but it has to be the *same*
// value channel OpenCV's `COLOR_BGR2HSV_FULL` / `COLOR_HSV2BGR_FULL` pair
// produces, because that is the plane the reference implementation corrects.
// The round trip is not an identity: the forward conversion quantises hue and
// saturation through its fixed-point tables and the inverse reconstructs the
// colour from those quantised values, so a colour can shift by a few levels
// even where the correction leaves its value channel alone. That loss is
// reproduced here rather than avoided, because the correction is defined in
// terms of the quantised plane.

/// Number of fractional bits of the forward conversion's lookup tables.
const HSV_SHIFT: u32 = 12;
/// Full hue range of the `*_FULL` conversions (`hrange = 256`).
const HRANGE_HSV: i32 = 256;
/// `hscale` of `cv::HSV2RGB_b` for full-range input (`6.0f / 255`).
const HSCALE: f32 = 6.0 / 255.0;

/// `hdiv_table256` of `cv::RGB2HSV_b`: `(256 << 12) / (6 * i)`, rounded.
fn hdiv_table() -> [i32; 256] {
    let mut table = [0i32; 256];
    for (i, entry) in table.iter_mut().enumerate().skip(1) {
        *entry = ((HRANGE_HSV << HSV_SHIFT) as f64 / (6.0 * i as f64)).round() as i32;
    }
    table
}

/// `sdiv_table` of `cv::RGB2HSV_b`: `(255 << 12) / i`, rounded.
fn sdiv_table() -> [i32; 256] {
    let mut table = [0i32; 256];
    for (i, entry) in table.iter_mut().enumerate().skip(1) {
        *entry = ((255i32 << HSV_SHIFT) as f64 / i as f64).round() as i32;
    }
    table
}

/// The per-pixel kernel of `cv::RGB2HSV_b`; `hsv` receives H, S, V.
fn rgb_pixel_to_hsv(
    r: f64,
    g: f64,
    b: f64,
    hdiv: &[i32; 256],
    sdiv: &[i32; 256],
    hsv: &mut [u8; 3],
) {
    let v = r.max(g).max(b);
    let vmin = r.min(g).min(b);

    let diff = (v - vmin) as u8 as usize;
    let vr = if v == r { -1i32 } else { 0 };
    let vg = if v == g { -1i32 } else { 0 };

    let s = (diff as i32 * sdiv[v as usize] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
    let mut h = (vr & (g - b) as i32)
        + (!vr
            & ((vg & (b - r + 2.0 * diff as f64) as i32)
                + (!vg & (r - g + 4.0 * diff as f64) as i32)));
    h = (h * hdiv[diff] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
    h += if h < 0 { HRANGE_HSV } else { 0 };

    hsv[0] = h.clamp(0, 255) as u8;
    hsv[1] = s.clamp(0, 255) as u8;
    hsv[2] = v as u8;
}

/// `cv::HSV2RGB_b` for a single pixel; returns B, G, R in `0..=255`.
///
/// Follows `cv::HSV2RGB_b` except that it truncates its channels instead of
/// rounding them through `cv::saturate_cast<uchar>`. The channel that carries
/// the value is an integer already, so only the two interpolated channels can
/// differ, by one level, and only where the exact value falls on a level. The
/// arithmetic runs in `f32`.
fn hsv_pixel_to_rgb(h: u8, s: u8, v: u8) -> [u8; 3] {
    const SECTOR_DATA: [[usize; 3]; 6] = [
        [1, 3, 0],
        [1, 0, 2],
        [3, 0, 1],
        [0, 2, 1],
        [0, 1, 3],
        [2, 1, 0],
    ];

    let v = v as f32;
    let s = s as f32 * (1.0 / 255.0);
    if s == 0.0 {
        let v = saturate_u8(v as f64);
        return [v, v, v];
    }

    // ComputeSectorAndClampedH(): h is scaled, split into a sector and then
    // clamped, so the fractional part can never leave 0..=1.
    let mut h = h as f32 * HSCALE;
    let sector_floor = h.floor();
    h -= sector_floor;
    let mut sector = sector_floor as i32 % 6;
    if sector < 0 {
        sector += 6;
    }
    let h = h.clamp(0.0, 1.0);

    let tab = [
        v,
        v * (1.0 - s),
        v * (1.0 - s * h),
        v * (1.0 - s * (1.0 - h)),
    ];
    let d = SECTOR_DATA[sector as usize];
    [tab[d[0]] as u8, tab[d[1]] as u8, tab[d[2]] as u8]
}

/// The value channel of an interleaved RGB image, as OpenCV's
/// `COLOR_BGR2HSV_FULL` computes it.
pub fn rgb_to_value_plane(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let hdiv = hdiv_table();
    let sdiv = sdiv_table();
    let mut value = vec![0u8; width * height];
    for (pixel, out) in value.iter_mut().enumerate() {
        let hsv = &mut [0u8; 3];
        rgb_pixel_to_hsv(
            rgb[3 * pixel] as f64,
            rgb[3 * pixel + 1] as f64,
            rgb[3 * pixel + 2] as f64,
            &hdiv,
            &sdiv,
            hsv,
        );
        *out = hsv[2];
    }
    value
}

/// Replace the value channel of an interleaved RGB image and convert back, as
/// `cv::COLOR_HSV2BGR_FULL` does.
pub fn value_plane_to_rgb(rgb: &[u8], value: &[u8], out: &mut [u8], width: usize, height: usize) {
    let hdiv = hdiv_table();
    let sdiv = sdiv_table();
    for pixel in 0..width * height {
        let (r, g, b) = (
            rgb[3 * pixel] as f64,
            rgb[3 * pixel + 1] as f64,
            rgb[3 * pixel + 2] as f64,
        );
        let mut hsv = [0u8; 3];
        rgb_pixel_to_hsv(r, g, b, &hdiv, &sdiv, &mut hsv);
        let bgr = hsv_pixel_to_rgb(hsv[0], hsv[1], value[pixel]);
        out[3 * pixel] = bgr[2];
        out[3 * pixel + 1] = bgr[1];
        out[3 * pixel + 2] = bgr[0];
    }
}

// ---------------------------------------------------------------------------
// The correction itself
// ---------------------------------------------------------------------------

/// `T_t`, the expected global average of the intensity plane, on the 0..=255
/// scale of the reference implementation.
pub const DEFAULT_TARGET: f64 = 112.0;
/// `tau_t`, the relative deviation from [`DEFAULT_TARGET`] that marks an image
/// as dimmed or bright.
pub const DEFAULT_TAU_T: f64 = 0.3;
/// `tau`, the floor of the inverse CDF used by the truncated (bright) path.
pub const DEFAULT_TAU: f64 = 0.5;
/// `alpha_dimmed`, the weighting exponent of the dimmed path.
pub const DEFAULT_ALPHA_DIMMED: f64 = 0.75;
/// `alpha_bright`, the weighting exponent of the bright path.
pub const DEFAULT_ALPHA_BRIGHT: f64 = 0.25;
/// Relative half-width of the "leave the image alone" band around the
/// thresholds.
///
/// A deviation within `|t| <= tau_t + DECISION_BAND` is `Unchanged`, which
/// keeps the decision stable: without the band a scene sitting on the threshold
/// would take a correcting path in one run and none of it in the next, after a
/// change too small to see.
pub const DECISION_BAND: f64 = 0.01;

/// Which correction an image needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The image is already near the target brightness: it is copied through.
    Unchanged,
    /// Dimmed below the target and below `-tau_t`: inverse gamma, no truncation.
    Dimmed,
    /// Brighter than the target and above `tau_t`: inverse gamma, truncated.
    Bright,
}

impl Mode {
    /// Whether the plane runs through `255 - L` before and after the LUT.
    ///
    /// Only the bright path inverts: its dark end is where the correction has
    /// room to expand, so the histogram, the weighting distribution and the
    /// table are all built from `255 - L`.
    pub fn is_inverted(self) -> bool {
        self == Mode::Bright
    }

    /// The name of the path, as reported by `--stats`.
    pub fn label(self) -> &'static str {
        match self {
            Mode::Unchanged => "Unchanged",
            Mode::Dimmed => "Dimmed",
            Mode::Bright => "Bright Image",
        }
    }
}

/// The parameters of the enhancement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// `T_t`, the expected global average of the intensity plane.
    pub target: f64,
    /// `tau_t`, the relative deviation that marks an image as dimmed or bright.
    pub tau_t: f64,
    /// `tau`, the floor of the inverse CDF on the truncated (bright) path.
    pub tau: f64,
    /// `alpha_dimmed`, the weighting exponent of the dimmed path.
    pub alpha_dimmed: f64,
    /// `alpha_bright`, the weighting exponent of the bright path.
    pub alpha_bright: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            target: DEFAULT_TARGET,
            tau_t: DEFAULT_TAU_T,
            tau: DEFAULT_TAU,
            alpha_dimmed: DEFAULT_ALPHA_DIMMED,
            alpha_bright: DEFAULT_ALPHA_BRIGHT,
        }
    }
}

impl Params {
    /// `(mean(L) - T_t) / T_t`, the relative deviation of the image's mean
    /// intensity from the target.
    pub fn deviation(&self, mean_intensity: f64) -> f64 {
        (mean_intensity - self.target) / self.target
    }

    /// Which correction the deviation calls for.
    ///
    /// `Unchanged` also covers a `+/-`[`DECISION_BAND`] neighbourhood of the
    /// thresholds; see [`DECISION_BAND`].
    pub fn mode(&self, deviation: f64) -> Mode {
        if deviation < -(self.tau_t + DECISION_BAND) {
            Mode::Dimmed
        } else if deviation > self.tau_t + DECISION_BAND {
            Mode::Bright
        } else {
            Mode::Unchanged
        }
    }

    /// The weighting exponent of a mode's path.
    pub fn alpha(&self, mode: Mode) -> f64 {
        match mode {
            Mode::Dimmed => self.alpha_dimmed,
            Mode::Bright => self.alpha_bright,
            Mode::Unchanged => f64::NAN,
        }
    }
}

/// What the pipeline decided and measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Info {
    /// The chosen correction.
    pub mode: Mode,
    /// Mean of the intensity plane, before any inversion, in `0..=255`.
    pub mean_intensity: f64,
    /// [`Params::deviation`] of that mean.
    pub deviation: f64,
    /// The weighting exponent of the chosen path.
    pub alpha: f64,
    /// Whether the inverse CDF was floored at `tau`.
    pub truncated: bool,
    /// Whether the corrected plane was inverted back after the LUT.
    pub inverted: bool,
    /// Distinct levels that received a gamma from a non-empty histogram bin,
    /// i.e. the size of the effective correction table.
    pub levels: usize,
    /// The table's value at level [`TAB_LOW`], the last entry that is pinned
    /// rather than computed.
    pub table_low: u8,
}

/// Floor of the `pdf_max - pdf_min` denominator of the weighting distribution.
///
/// With every occupied level holding the same probability that difference is
/// zero: flooring it at `1 / MIN_PROB` spreads the distribution over `1 /
/// MIN_PROB` levels, which keeps the table finite and still strictly
/// increasing, and is all the mapping needs to be a gamma curve.
const MIN_PROB: f64 = 1e-8;

/// The first table level whose gamma is computed rather than forced.
///
/// Level 0 is pinned to itself, so black stays black: with `gamma[0] = 1` the
/// formula gives `255 * (0/255)^1 = 0` too, but forcing the entry keeps the
/// fixed point exact instead of relying on a `0` raised to a power.
pub const TAB_LOW: usize = 1;

/// `cv::saturate_cast<uchar>`: round half to even, then clamp.
pub fn saturate_u8(x: f64) -> u8 {
    x.round_ties_even().clamp(0.0, 255.0) as u8
}

/// The empirical probability mass function of an 8-bit intensity plane.
pub fn histogram(data: &[u8]) -> [f64; 256] {
    let mut hist = [0.0f64; 256];
    for sample in data {
        hist[*sample as usize] += 1.0;
    }
    let inv = 1.0 / data.len().max(1) as f64;
    for bin in hist.iter_mut() {
        *bin *= inv;
    }
    hist
}

/// Apply the correction to an intensity plane, in place.
///
/// The mode is not decided here - [`Params::mode`] does that from the plane's
/// mean - so that a caller can report the decision before the correction runs.
/// `mode == Mode::Unchanged` is a no-op.
pub fn correct_intensity(data: &mut [u8], params: &Params, mode: Mode) -> Info {
    let mean_intensity = data.iter().map(|s| *s as f64).sum::<f64>() / data.len().max(1) as f64;
    let deviation = params.deviation(mean_intensity);
    let mut info = Info {
        mode,
        mean_intensity,
        deviation,
        alpha: params.alpha(mode),
        truncated: false,
        inverted: false,
        levels: 0,
        table_low: 0,
    };
    if mode == Mode::Unchanged {
        return info;
    }

    info.truncated = mode == Mode::Bright;
    info.inverted = mode.is_inverted();
    if info.inverted {
        for sample in data.iter_mut() {
            *sample = 255 - *sample;
        }
    }

    let table = correction_table(&histogram(data), params, mode);
    info.levels = table.levels;
    info.table_low = table.values[TAB_LOW];
    for sample in data.iter_mut() {
        *sample = table.values[*sample as usize];
    }

    if info.inverted {
        for sample in data.iter_mut() {
            *sample = 255 - *sample;
        }
    }
    info
}

/// The 256-entry correction table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Table {
    pub values: [u8; 256],
    /// Number of distinct levels with a non-zero probability, i.e. the levels
    /// the gamma was computed from rather than interpolated around.
    pub levels: usize,
}

/// Build the correction table of the weighting distribution.
///
/// Levels whose histogram bin is empty have a zero weight and would follow
/// `gamma = 1`: the identity. They therefore sit exactly at the value of the
/// nearest occupied level below them - the correction is a constant extension
/// there, not a curve.
pub fn correction_table(pdf: &[f64; 256], params: &Params, mode: Mode) -> Table {
    let alpha = params.alpha(mode);
    let pdf_min = pdf.iter().copied().fold(f64::INFINITY, f64::min);
    let pdf_max = pdf.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // A flat sample range would divide by zero here, so the denominator is
    // floored at `1 / MIN_PROB`: see [`MIN_PROB`].
    let spread = (pdf_max - pdf_min).max(MIN_PROB);
    let scale = pdf_max / spread;

    let mut weights = [0.0f64; 256];
    // A level with no mass keeps a zero weight and is filled in from its
    // nearest occupied neighbour below afterwards.
    let mut gamma = [f64::NAN; 256];
    let mut occupied = Vec::with_capacity(256);
    let mut total = 0.0f64;
    for level in 0..256 {
        let normalized = (pdf[level] - pdf_min) / spread;
        if normalized > 0.0 {
            weights[level] = scale * normalized.powf(alpha);
            total += weights[level];
            gamma[level] = 0.0;
            occupied.push(level);
        }
    }
    if total > 0.0 {
        let mut cumsum = 0.0f64;
        for level in &occupied {
            cumsum += weights[*level];
            let cdf = cumsum / total;
            gamma[*level] = match mode {
                Mode::Bright => (1.0 - cdf).max(params.tau),
                _ => 1.0 - cdf,
            };
        }
    }

    let mut values = [0u8; 256];
    for level in TAB_LOW..256 {
        // An unoccupied level has no gamma of its own; it rounds the level
        // below, which leaves the curve constant across a gap rather than
        // folding it back down.
        let exponent = if gamma[level].is_nan() {
            gamma[level - 1]
        } else {
            gamma[level]
        };
        values[level] = saturate_u8(255.0 * (level as f64 / 255.0).powf(exponent));
    }

    Table {
        values,
        levels: occupied.len(),
    }
}

/// Run the enhancement on an RGB image; `rgb` holds `width * height * 3`
/// interleaved bytes and the returned buffer has the same layout.
pub fn enhance_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    params: &Params,
    mode: Option<Mode>,
) -> (Vec<u8>, Info) {
    assert!(width > 0 && height > 0, "empty image");
    assert_eq!(rgb.len(), width * height * 3, "rgb buffer size mismatch");

    let mut value = rgb_to_value_plane(rgb, width, height);
    let mode = mode.unwrap_or_else(|| params.mode(params.deviation(mean(&value))));
    let info = correct_intensity(&mut value, params, mode);
    let mut out = vec![0u8; width * height * 3];
    value_plane_to_rgb(rgb, &value, &mut out, width, height);
    (out, info)
}

/// Run the enhancement on a single-channel intensity image, in place.
///
/// `data` holds `width * height` samples.
pub fn enhance_gray(
    data: &mut [u8],
    width: usize,
    height: usize,
    params: &Params,
    mode: Option<Mode>,
) -> Info {
    assert!(width > 0 && height > 0, "empty image");
    assert_eq!(data.len(), width * height, "intensity buffer size mismatch");
    let mode = mode.unwrap_or_else(|| params.mode(params.deviation(mean(data))));
    correct_intensity(data, params, mode)
}

/// Mean of an intensity plane, defaulting to `0.0` when it is empty.
fn mean(data: &[u8]) -> f64 {
    data.iter().map(|s| *s as f64).sum::<f64>() / data.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The correction is constant on every level the image does not contain, so
    /// only the levels that are present carry a gamma of their own.
    fn table_on(texture: &[u8], params: &Params, mode: Mode) -> [u8; 256] {
        let mut plane = texture.to_vec();
        correct_intensity(&mut plane, params, mode);
        correction_table(&histogram(&plane), params, mode).values
    }

    #[test]
    fn table_is_monotone_and_anchored() {
        let mut texture = Vec::new();
        for x in 0..64 {
            for y in 0..64 {
                texture.push((x * 3 + y * 5) as u8);
            }
        }
        for mode in [Mode::Dimmed, Mode::Bright] {
            let table = table_on(&texture, &Params::default(), mode);
            assert_eq!(table[0], 0, "{mode:?}: black stays black");
            assert_eq!(table[255], 255, "{mode:?}: white stays white");
            assert!(table.windows(2).all(|w| w[0] <= w[1]), "{mode:?}: monotone");
        }
    }

    #[test]
    fn modes_follow_the_thresholds() {
        let params = Params::default();
        assert_eq!(params.mode(0.0), Mode::Unchanged);
        assert_eq!(params.mode(-0.29), Mode::Unchanged);
        assert_eq!(params.mode(0.29), Mode::Unchanged);
        // Inside the band.
        assert_eq!(params.mode(-0.3), Mode::Unchanged);
        assert_eq!(params.mode(0.3), Mode::Unchanged);
        assert_eq!(params.mode(-0.305), Mode::Unchanged);
        assert_eq!(params.mode(0.305), Mode::Unchanged);
        // Outside it, on the thresholds themselves.
        assert_eq!(params.mode(-0.311), Mode::Dimmed);
        assert_eq!(params.mode(0.311), Mode::Bright);
        assert_eq!(params.mode(-0.486), Mode::Dimmed);
        assert_eq!(params.mode(0.5), Mode::Bright);
    }

    /// The three paths, on a plane whose mean puts it in the middle of the band.
    #[test]
    fn paths_differ_in_inversion_and_truncation() {
        let params = Params::default();

        let mut dimmed = vec![40u8; 16 * 16];
        for (i, value) in dimmed.iter_mut().enumerate() {
            *value = (40 + i % 40) as u8;
        }
        let info = correct_intensity(&mut dimmed, &params, Mode::Dimmed);
        assert!(!info.truncated);
        assert!(!info.inverted);
        assert!(info.alpha == params.alpha_dimmed);

        let mut bright: Vec<u8> = (0..16 * 16).map(|i| (200 + i % 40) as u8).collect();
        let info = correct_intensity(&mut bright, &params, Mode::Bright);
        assert!(info.truncated);
        assert!(info.inverted);
        assert!(info.alpha == params.alpha_bright);

        let mut untouched = vec![0u8, 255, 100, 100];
        let before = untouched.clone();
        let info = correct_intensity(&mut untouched, &params, Mode::Unchanged);
        assert_eq!(untouched, before);
        assert_eq!(info.levels, 0);
    }

    #[test]
    fn mean_is_measured_on_the_value_plane() {
        let rgb = vec![255u8, 0, 0, 10, 20, 30];
        let (_, info) = enhance_rgb(&rgb, 2, 1, &Params::default(), None);
        // The value channel of (255, 0, 0) is 255 and of (10, 20, 30) is 30.
        assert!((info.mean_intensity - 142.5).abs() < 1e-12);
    }

    #[test]
    fn dimmed_image_is_brightened() {
        let mut plane: Vec<u8> = (0..64 * 64).map(|i| (i % 64) as u8).collect();
        let before = mean(&plane);
        let info = correct_intensity(&mut plane, &Params::default(), Mode::Dimmed);
        assert!(!info.truncated);
        assert!(!info.inverted);
        assert!(mean(&plane) > before, "dimmed path brightens");
    }

    #[test]
    fn a_degenerate_plane_stays_finite() {
        // Every sample identical: the weighting distribution is flat.
        for mode in [Mode::Dimmed, Mode::Bright] {
            let mut plane = vec![130u8; 32 * 32];
            correct_intensity(&mut plane, &params(), mode);
            assert!(plane.iter().all(|s| *s == plane[0]), "{mode:?}: constant");
        }
    }

    fn params() -> Params {
        Params::default()
    }

    /// The value channel of a pixel is its largest sample, exactly.
    #[test]
    fn value_is_the_maximum_sample() {
        let rgb: Vec<u8> = vec![10, 200, 30, 255, 0, 0, 0, 0, 0, 7, 7, 7];
        let value = rgb_to_value_plane(&rgb, 4, 1);
        assert_eq!(value, vec![200, 255, 0, 7]);
    }

    /// A grey image is invariant under the round trip.
    #[test]
    fn grey_round_trips() {
        let rgb: Vec<u8> = (0..=255u8).flat_map(|g| [g, g, g]).collect();
        let value = rgb_to_value_plane(&rgb, 256, 1);
        assert_eq!(value, (0..=255u8).collect::<Vec<_>>());
        let mut out = vec![0u8; rgb.len()];
        value_plane_to_rgb(&rgb, &value, &mut out, 256, 1);
        assert_eq!(out, rgb);
    }

    /// The round trip reproduces the reference's quantisation of hue and
    /// saturation.
    #[test]
    fn round_trip_matches_reference_quantisation() {
        let rgb: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
        let value = rgb_to_value_plane(&rgb, 4, 1);
        assert_eq!(value, vec![255, 255, 255, 255]);
        let mut out = vec![0u8; rgb.len()];
        value_plane_to_rgb(&rgb, &value, &mut out, 4, 1);
        assert_eq!(out, vec![255, 0, 0, 0, 255, 0, 6, 0, 255, 251, 255, 0]);
    }

    /// The interpolated channels are truncated, and in `f32` that can be a
    /// level below the exact expression.
    #[test]
    fn interpolated_channels_are_truncated() {
        // Sector 3, hue 136 of 256 and saturation 55 of 255. In `B, G, R`:
        // blue is the value itself, green is `153 * (1 - 0.2 * 55 / 255)`,
        // i.e. `146.4`, and red is `153 * (1 - 55 / 255)`, which is `120`
        // exactly but evaluates to `119.99999` in `f32` and truncates to `119`.
        assert_eq!(hsv_pixel_to_rgb(136, 55, 153), [153, 146, 119]);
        // A grey pixel bypasses the tab entirely.
        assert_eq!(hsv_pixel_to_rgb(0, 0, 200), [200, 200, 200]);
    }
}
