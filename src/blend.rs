//! The blending rule: what the synthesised exposure is allowed to contribute.
//!
//! The fusion reduces to three ingredients: the normalised input `I`, an
//! illumination map `t`, and a synthesised exposure `J` from one global camera
//! response curve. The usual rule is the per-pixel mix
//!
//! ```text
//! out = I * t^mu + J * (1 - t^mu)          mu = 0.5
//! ```
//!
//! which destroys highlight tone, because `J` is not confined to `[0, 1]`: with
//! the camera response in [`crate::fusion`], the sample at `I = 1` lands at
//! `J = 1.58..1.69` for every exposure ratio the entropy search chooses, so the
//! brightest part of a photograph carries its tone in `J`'s out-of-range slope.
//! Because the output is an 8-bit sample, any such mix clamps that slope away.
//! This module maps `J` instead of mixing it, in four steps:
//!
//! 1. **Polarity.** Which candidate carries the pixel is decided by the
//!    illumination map alone: `W = 1 - t`, so `out = I * (1 - t) + mapped * t`.
//!    A sample the estimator calls well lit (`t -> 1`) is replaced by the
//!    synthesised exposure; a dark one (`t -> 0`) keeps the original.
//! 2. **A tone map that does not clip.** `J`'s luminance goes through
//!    `g(j) = j` below a knee and, above it, the straight line from
//!    `(knee, knee)` to `(J_max, CEILING)`:
//!
//!    ```text
//!    g(j) = knee + (CEILING - knee) * (j - knee) / (J_max - knee)     j > knee
//!    ```
//!
//!    `CEILING` is `249/255`, four places below white so the clipped and the
//!    merely bright stay distinguishable, and the slope above the knee is
//!    always well below 1. Below the knee the rule is exactly
//!    `out = I * (1 - t) + J * t`.
//! 3. **Colour above the knee.** Compressing luminance alone would raise chroma,
//!    because a channel at a fixed offset over its luminance becomes a larger
//!    fraction of a compressed luminance, so the channels are rebuilt around the
//!    mapped luminance - `mapped_c = g(L) + s * (J_c - L)` with `L` the
//!    luminance of `J` - and `s` is bounded twice: by the ceiling,
//!    `room = (CEILING - g(L)) / spread`, the largest scale that keeps the
//!    pixel's brightest channel under `CEILING`; and by the input itself,
//!    `cap = input_spread / synthetic_spread`, the scale at which the pixel has
//!    exactly the chroma the input had. `s = min(room, cap)`, and 1.0 below the
//!    knee.
//! 4. **The feather.** `cap` is a condition - it binds or it does not - so
//!    switching it on at the knee would put a step in the output there. The
//!    repair ramps the *argument* rather than the output, over a fraction of the
//!    headroom:
//!
//!    ```text
//!    f = FEATHER_FRACTION * (J_max - knee)     band width, synthetic units
//!    t = clamp((j - knee) / f, 0, 1)
//!    w = t * t * (3 - 2t)                      smoothstep: zero slope at both ends
//!    s = min(room, 1 + (cap - 1) * w)
//!    ```
//!
//!    The cap term passes through 1 at the knee and reaches the full cap
//!    smoothly, so the join is C1 and the map - a `min` of two continuous
//!    functions - has no seam. The ramp is in `j`, a property of the pixel, so
//!    it is not a spatial blur and cannot invent structure.
//!
//! `knee` is the one exposed parameter
//! ([`crate::fusion::FusionParams::knee`]): it is the point where the rule stops
//! being the plain polarised mix, it trades frame brightness against mid-tone
//! contrast almost linearly, and it is a taste call. `FEATHER_FRACTION` is not
//! exposed: it is the width at which the seam stops being visible, and it has no
//! whole-frame effect at all.

/// Top of the mapped range, `249/255`.
///
/// Four places below 255 is deliberate: the map must not land on white, or the
/// clipped and the merely bright become indistinguishable.
pub const CEILING: f64 = 249.0 / 255.0;

/// Width of the chroma-cap feather, as a fraction of the headroom
/// `J_max - knee`.
pub const FEATHER_FRACTION: f64 = 0.113;

/// The per-pixel inputs of the rule.
///
/// Everything is already normalised: `input` and `synthetic` are RGB
/// interleaved values in `[0, 1]` (the synthesised exposure may exceed 1), and
/// `illumination` is the estimator's map, one value in `[0, 1]` per pixel.
pub struct BlendContext<'a> {
    /// The normalised input, RGB interleaved.
    pub input: InputView<'a>,
    /// Synthesised exposure, RGB interleaved, already carrying the camera
    /// model's offset.
    pub synthetic: &'a [f64],
    /// Illumination map, `width * height` values in `[0, 1]`.
    pub illumination: &'a [f64],
    pub width: usize,
    pub height: usize,
}

/// The normalised input, either already expanded or read from the 8-bit source.
///
/// A caller that holds the source bytes and their normalisation can pass those
/// instead of a full `f64` copy of the input; `(v - min) / range` is the same
/// value the normalisation produced.
#[derive(Clone, Copy)]
pub enum InputView<'a> {
    /// Already-normalised samples, one per channel.
    Expanded(&'a [f64]),
    /// The 8-bit source and the normalisation it went through.
    Bytes { rgb: &'a [u8], min: f64, range: f64 },
}

impl InputView<'_> {
    /// Number of interleaved samples.
    fn len(&self) -> usize {
        match self {
            InputView::Expanded(values) => values.len(),
            InputView::Bytes { rgb, .. } => rgb.len(),
        }
    }

    /// The normalised value of one channel.
    #[inline]
    fn value(&self, index: usize) -> f64 {
        match self {
            InputView::Expanded(values) => values[index],
            InputView::Bytes { rgb, min, range } => (rgb[index] as f64 - min) / range,
        }
    }
}

/// RGB luminance, the quantity the tone map acts on.
fn luma3(rgb: &[f64], pixel: usize) -> f64 {
    let base = 3 * pixel;
    0.299 * rgb[base] + 0.587 * rgb[base + 1] + 0.114 * rgb[base + 2]
}

/// Apply the rule. `knee` is the point where the map stops being the identity.
pub fn blend(ctx: &BlendContext<'_>, knee: f64) -> Vec<u8> {
    let n = ctx.width * ctx.height;
    assert_eq!(ctx.input.len(), n * 3, "input buffer size mismatch");
    assert_eq!(ctx.synthetic.len(), n * 3, "synthetic buffer size mismatch");
    assert_eq!(
        ctx.illumination.len(),
        n,
        "illumination buffer size mismatch"
    );

    let j_max = ctx
        .synthetic
        .iter()
        .fold(0.0f64, |acc, v| acc.max(*v))
        .max(knee + 1e-6);
    let span = j_max - knee;
    let feather = FEATHER_FRACTION * span;

    let mut out = vec![0u8; n * 3];

    // Every pixel reads only shared inputs and writes its own three output
    // bytes, so splitting the output into chunks that are whole pixels cannot
    // change a value.
    let workers = crate::worker_count();
    if workers > 1 && n >= (1 << 15) * workers {
        let chunk = n.div_ceil(workers) * 3;
        let mut first = 0usize;
        std::thread::scope(|scope| {
            for block in out.chunks_mut(chunk) {
                let start = first;
                first += block.len() / 3;
                scope.spawn(move || {
                    for index in 0..block.len() / 3 {
                        blend_pixel(
                            ctx,
                            knee,
                            feather,
                            span,
                            start + index,
                            &mut block[3 * index..3 * index + 3],
                        );
                    }
                });
            }
        });
        return out;
    }

    for pixel in 0..n {
        blend_pixel(
            ctx,
            knee,
            feather,
            span,
            pixel,
            &mut out[3 * pixel..3 * pixel + 3],
        );
    }
    out
}

/// One pixel of the rule, writing its three output bytes.
#[allow(clippy::too_many_arguments)]
#[inline]
fn blend_pixel(
    ctx: &BlendContext<'_>,
    knee: f64,
    feather: f64,
    span: f64,
    pixel: usize,
    out: &mut [u8],
) {
    let shape = |j: f64| {
        if j <= knee {
            j
        } else {
            knee + (CEILING - knee) * (j - knee) / span
        }
    };

    let l = luma3(ctx.synthetic, pixel);
    let g = shape(l);
    let spread = (0..3).fold(0.0f64, |acc, c| acc.max(ctx.synthetic[3 * pixel + c] - l));
    let input_rgb = [
        ctx.input.value(3 * pixel),
        ctx.input.value(3 * pixel + 1),
        ctx.input.value(3 * pixel + 2),
    ];
    let input_l = luma3(&input_rgb, 0);
    let input_spread = (0..3).fold(0.0f64, |acc, c| acc.max(input_rgb[c] - input_l));

    // Both bounds are ratios of spreads, so a grey pixel (no spread at all)
    // has no chroma to scale and `s = 1` is exact there rather than a
    // division by zero.
    let (room, cap) = if spread > 1e-9 {
        (
            ((CEILING - g) / spread).clamp(0.0, 1.0),
            (input_spread / spread).clamp(0.0, 1.0),
        )
    } else {
        (1.0, 1.0)
    };

    let t = ((l - knee) / feather.max(1e-9)).clamp(0.0, 1.0);
    let w = t * t * (3.0 - 2.0 * t);
    let scale = room.min(1.0 + (cap - 1.0) * w);

    // `W = 1 - t`: the synthesised exposure carries the sample where the
    // estimator says it is well lit.
    let weight = (1.0 - ctx.illumination[pixel]).clamp(0.0, 1.0);
    for (channel, value) in out.iter_mut().enumerate() {
        let i = 3 * pixel + channel;
        let mapped = g + scale * (ctx.synthetic[i] - l);
        let mixed = input_rgb[channel] * weight + mapped * (1.0 - weight);
        *value = (mixed * 255.0).clamp(0.0, 255.0) as u8;
    }
}
