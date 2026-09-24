//! Automatic white balance for photographs shot on a white background.
//!
//! A product photo on a white sweep is supposed to have a neutral white
//! background, but the camera and the light leave a cast in it, and every
//! colour in the frame inherits that cast. This module measures the cast from
//! the background itself and removes it by scaling each channel so that the
//! background's white lands on 255.
//!
//! The white reference comes from the **outer edge** of the image, where the
//! background is: the border strip is 5% of the shorter side, its brightest 1%
//! by maximum channel is taken as a white reference, and the per-channel means
//! of those pixels are the white point. Reading only the border keeps the
//! subject out of the estimate; reading only the brightest 1% keeps a shadow on
//! the sweep out of it too.
//!
//! Two corrections are available:
//!
//! * **plain** ([`Params::default`]): `out = v * 255 / white_c`, clipped at 255.
//!   Every level is moved by the same factor.
//! * **soft** ([`Params::safe`]): the same per-channel gain on the
//!   white-normalised value, followed by a soft knee that leaves the low end of
//!   the range alone and rolls the top end asymptotically into 1.0, so a
//!   specular highlight or a very bright detail keeps its shape instead of
//!   clipping. The knee is a fraction of the white point, so it moves with the
//!   background rather than sitting at a fixed level.
//!
//! Both are applied through one lookup curve per channel, which is built once
//! and made monotone before it is used: a correction must never put a step in
//! the output that was not in the input, and a curve written level by level
//! can. Output levels are truncated rather than rounded, matching the reference
//! implementation this replaced.

use crate::png_io::{self, Image};

/// Fraction of the shorter side used as the edge strip, in percent.
///
/// This is the `min(h, w) // 20` of the reference implementation.
pub const EDGE_PERCENT: f64 = 5.0;

/// Percentage of the edge pixels taken as the white reference (`percentile(99)`).
pub const WHITE_PERCENT: f64 = 1.0;

/// Default soft-knee point, as a fraction of the white point.
pub const DEFAULT_KNEE: f64 = 0.5;

/// The white point measured for one image, on the 0..=255 scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhitePoint {
    /// Per-channel mean of the brightest edge pixels.
    pub channels: [f64; 3],
    /// `255 / channels`, the gain of each channel.
    pub gains: [f64; 3],
}

/// What the correction measured and did, for `--stats` and for tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Info {
    /// The white point the correction was built from.
    pub white: WhitePoint,
    /// Fraction of output samples that reached 255.
    pub clipped_fraction: f64,
}

/// The parameters of the correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Compress the highlights instead of clipping them.
    pub safe: bool,
    /// Soft-knee point as a fraction of the white point, used by the soft mode.
    pub knee: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            safe: false,
            knee: DEFAULT_KNEE,
        }
    }
}

impl Params {
    /// The parameters of the soft mode, whose knee is clamped to `0..=1`.
    pub fn safe(knee: f64) -> Self {
        Params {
            safe: true,
            knee: knee.clamp(0.0, 1.0),
        }
    }
}

/// The soft knee `k(x)`: the identity up to `knee`, then
/// `knee + room * (1 - exp(-(x - knee) / room))` with `room = 1 - knee`.
///
/// `x` is a fraction of the white point, so `x = 1` is the background. The two
/// pieces meet at the knee with the same value and the same slope, and the
/// exponential approaches 1.0 without reaching it, so nothing above the knee is
/// ever clipped.
#[inline]
pub fn soft_knee(x: f64, knee: f64) -> f64 {
    if x <= knee {
        return x;
    }
    let room = 1.0 - knee;
    if room <= 0.0 {
        return knee;
    }
    knee + room * (1.0 - (-(x - knee) / room).exp())
}

impl WhitePoint {
    /// Measure the white point of an image from its brightest border pixels.
    pub fn measure(image: &Image) -> WhitePoint {
        let samples: Vec<[u8; 3]> = image
            .rgb
            .chunks_exact(3)
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        let channels = border_white_point(&samples, image.width, image.height);
        WhitePoint {
            channels,
            gains: [
                255.0 / channels[0],
                255.0 / channels[1],
                255.0 / channels[2],
            ],
        }
    }
}

/// The per-channel white reference of the border pixels, on the 0..=255 scale.
///
/// Every channel is floored at 1 so a background with no white anywhere in the
/// strip divides by a sane number instead of by zero.
fn border_white_point(samples: &[[u8; 3]], width: usize, height: usize) -> [f64; 3] {
    let (width, height) = (width.max(1), height.max(1));
    // `min(h, w) // 20`, i.e. 5% of the shorter side, at least one pixel.
    let border = (width.min(height) / 20).max(1);

    let mut edge: Vec<[u8; 3]> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if y < border || y + border >= height || x < border || x + border >= width {
                if let Some(pixel) = samples.get(y * width + x) {
                    edge.push(*pixel);
                }
            }
        }
    }
    if edge.is_empty() {
        return [255.0; 3];
    }

    // Luminance as `max(r, g, b)` per pixel: the white reference is the
    // brightest edge pixels by their largest channel.
    let luminance = |pixel: &[u8; 3]| pixel[0].max(pixel[1]).max(pixel[2]) as f64;
    let mut order: Vec<usize> = (0..edge.len()).collect();
    order.sort_by(|a, b| luminance(&edge[*a]).total_cmp(&luminance(&edge[*b])));

    // `lum >= percentile(lum, 99)`: the brightest 1%, taken as a rank. Ties on
    // the cut value are included, which is what `>=` does on the interpolated
    // percentile the reference used.
    let total = edge.len();
    let rank = (((total as f64) * (100.0 - WHITE_PERCENT) / 100.0).floor() as usize).min(total - 1);
    let cut = luminance(&edge[order[rank]]);
    let mut sums = [0.0f64; 3];
    let mut count = 0usize;
    for index in order {
        if luminance(&edge[index]) >= cut {
            for (channel, sum) in sums.iter_mut().enumerate() {
                *sum += edge[index][channel] as f64;
            }
            count += 1;
        }
    }
    let count = count.max(1) as f64;
    [
        (sums[0] / count).max(1.0),
        (sums[1] / count).max(1.0),
        (sums[2] / count).max(1.0),
    ]
}

/// One channel's correction curve: the output level for each of the 256 inputs.
///
/// `white` is that channel's white point on the 0..=255 scale. The curve is
/// forced to be non-decreasing before it is returned. The two rules are already
/// monotone, so the pass is a no-op on a clean image; it is there so that a
/// level the correction cannot move cannot end up above one it can, which would
/// invert two neighbouring levels of a gradient.
pub fn curve(white: f64, params: &Params) -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut previous = 0u8;
    for (level, out) in table.iter_mut().enumerate() {
        let source = level as f64;
        let target = if params.safe {
            // The knee is a fraction of the white point, so the input is
            // normalised by the white point alone and scaled back afterwards.
            soft_knee(source / white, params.knee) * 255.0
        } else {
            source * (255.0 / white)
        };
        // Truncating, as the reference's `astype(np.uint8)` did; a float cast
        // saturates, so the clip is already implied.
        let value = target.clamp(0.0, 255.0) as u8;
        previous = previous.max(value);
        *out = previous;
    }
    table
}

/// Correct an image in place, returning what was measured.
pub fn correct_image(image: &mut Image, params: &Params) -> Info {
    let white = WhitePoint::measure(image);
    let pixels = image.width * image.height;

    let tables = [
        curve(white.channels[0], params),
        curve(white.channels[1], params),
        curve(white.channels[2], params),
    ];

    let mut clipped = 0usize;
    let rgb: &mut [u8] = image.rgb.as_mut();
    for pixel in 0..pixels {
        let start = 3 * pixel;
        for (value, table) in rgb[start..start + 3].iter_mut().zip(&tables) {
            *value = table[*value as usize];
            clipped += usize::from(*value == 255);
        }
    }

    Info {
        white,
        clipped_fraction: clipped as f64 / (pixels * 3).max(1) as f64,
    }
}

/// Decode a PNG, correct it and re-encode it as a PNG.
pub fn white_balance_png(input: &[u8], params: &Params) -> Result<Vec<u8>, String> {
    let mut image = png_io::decode_png(input)?;
    correct_image(&mut image, params);
    png_io::encode_png(&image)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An image whose border is a flat tinted white and whose centre is
    /// saturated, so the white point is the border and nothing else.
    fn tinted(width: usize, height: usize) -> Image {
        let mut rgb = vec![0u8; width * height * 3];
        for pixel in 0..width * height {
            rgb[3 * pixel] = 200;
            rgb[3 * pixel + 1] = 190;
            rgb[3 * pixel + 2] = 175;
        }
        for y in 10..height - 10 {
            for x in 10..width - 10 {
                let at = 3 * (y * width + x);
                rgb[at] = 20;
                rgb[at + 1] = 90;
                rgb[at + 2] = 200;
            }
        }
        Image {
            width,
            height,
            rgb,
            alpha: None,
        }
    }

    #[test]
    fn white_point_is_the_border() {
        let image = tinted(120, 80);
        let white = WhitePoint::measure(&image);
        assert!((white.channels[0] - 200.0).abs() < 1e-9, "{white:?}");
        assert!((white.channels[1] - 190.0).abs() < 1e-9, "{white:?}");
        assert!((white.channels[2] - 175.0).abs() < 1e-9, "{white:?}");
    }

    #[test]
    fn the_plain_mode_takes_the_background_to_white() {
        let mut image = tinted(120, 80);
        let info = correct_image(&mut image, &Params::default());
        // The red channel lands one level short of 255 because `200 * (255 /
        // 200)` is a hair under 255 in `f64` and the output truncates, which is
        // exactly what the reference implementation produced as well.
        assert_eq!(&image.rgb[..3], &[254, 255, 255], "{info:?}");
        // The subject keeps its hue: blue stays the strongest channel, and it
        // is the only channel of the subject that reaches the clip.
        let at = 3 * (60 * 120 + 60);
        assert_eq!(&image.rgb[at..at + 3], &[25, 120, 255], "{info:?}");
    }

    #[test]
    fn the_plain_mode_never_darkens_a_level() {
        let mut image = tinted(120, 80);
        correct_image(&mut image, &Params::default());
        let before = tinted(120, 80).rgb;
        for (after, before) in image.rgb.iter().zip(before) {
            assert!(*after >= before, "{before} -> {after}");
        }
    }

    #[test]
    fn the_soft_mode_never_reaches_255() {
        let mut image = tinted(120, 80);
        let info = correct_image(&mut image, &Params::safe(0.5));
        assert_eq!(info.clipped_fraction, 0.0, "{info:?}");
        // The border sits exactly on the knee, so it lands just above it
        // rather than on 255: the knee is a fraction of the white point, not
        // of the range.
        assert_eq!(&image.rgb[..3], &[208, 208, 208], "{info:?}");
    }

    #[test]
    fn a_white_image_is_left_alone_by_the_plain_mode() {
        let mut image = Image {
            width: 4,
            height: 4,
            rgb: vec![255; 4 * 4 * 3],
            alpha: None,
        };
        correct_image(&mut image, &Params::default());
        assert_eq!(image.rgb, vec![255; 4 * 4 * 3]);
    }

    #[test]
    fn a_curve_never_decreases() {
        for safe in [false, true] {
            for white in [1.0, 64.0, 128.0, 200.0, 255.0] {
                for knee in [0.0, 0.25, 0.5, 0.85, 1.0] {
                    let table = curve(white, &Params { safe, knee });
                    for level in 1..256 {
                        assert!(
                            table[level] >= table[level - 1],
                            "white {white} safe {safe} knee {knee} at {level}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn soft_knee_is_continuous_at_the_knee() {
        let knee = 0.5;
        let below = soft_knee(knee - 1e-9, knee);
        let above = soft_knee(knee + 1e-9, knee);
        assert!((below - above).abs() < 1e-6, "{below} vs {above}");
        assert!(soft_knee(1.0, knee) < 1.0);
        assert_eq!(soft_knee(0.25, knee), 0.25);
    }
}
