//! Adaptive image enhancement.
//!
//! A standalone Rust implementation of the following contrast enhancement:
//!
//! ```text
//! RGB -> HSV_FULL, split
//! V_g  = mean of three vertical 5x1 Gaussian blurs of V (sigma 15, 80, 250)
//! V1   = (255 + k1) * V / (max(V, V_g) + k1)     k1 = 0.1 * mean(S)
//! V2   = (255 + k2) * V / (max(V, V_g) + k2)     k2 =      mean(S)
//! F    = w1 * V1 + w2 * V2   (w from the principal eigenvector of the 2x2
//!                             covariance of (V1, V2) over all pixels)
//! HSV_FULL -> RGB with V replaced by round(F)
//! ```
//!
//! The arithmetic follows OpenCV, which the reference implementation is built
//! on: the 8-bit `RGB2HSV_FULL` conversion uses OpenCV's fixed-point integer
//! tables, the 8-bit `HSV2RGB_FULL` conversion its `f32` path with
//! round-half-to-even, and the blurs, ratios and covariance are computed in
//! `f64`. The eigenvector of the 2x2 covariance is computed in closed form
//! rather than by `cv::eigen`; `w1` is invariant under the sign of the
//! eigenvector, and the closed form agrees with `cv::eigen` to ~1e-11 in `f64`,
//! far below the 1/255 quantisation step of `F`.
//!
//! # Exposure fusion
//!
//! The crate also implements the exposure fusion framework around that
//! enhancement (CAIP 2017). The adaptive enhancement is the illumination
//! estimator, used on its own or as the first stage of the fusion:
//!
//! * [`fusion`] - the camera response model, the entropy-optimal exposure
//!   ratio, and the fusion itself;
//! * [`blend`] - the highlight map that decides how much of the synthesised
//!   exposure is used;
//! * [`pipeline`] - the end-to-end entry points;
//! * [`resize`] - the `area` and `bicubic` resampling the fusion needs.

pub mod blend;
pub mod fusion;
pub mod pipeline;
pub mod png_io;
pub mod resize;

/// Kernel size of the three Gaussian blurs (`ksize = 5` in the original).
const KSIZE: usize = 5;
/// Standard deviations of the three Gaussian blurs (15, 80, 250).
const SIGMAS: [f64; 3] = [15.0, 80.0, 250.0];

/// Number of fractional bits used by OpenCV's 8-bit RGB->HSV lookup tables.
const HSV_SHIFT: u32 = 12;
/// Full hue range of the `*_FULL` colour conversions (OpenCV: `hrange = 256`).
const HRANGE_HSV: i32 = 256;
/// `HSV2RGB_b::hscale` for full-range input (OpenCV: `6.0f / 255`).
const HSCALE: f32 = 6.0 / 255.0;

// ---------------------------------------------------------------------------
// RGB <-> HSV (full range, 8 bit), ported from OpenCV
// ---------------------------------------------------------------------------

/// `hdiv_table256` of `cv::RGB2HSV_b`: `saturate_cast<int>((256 << 12) / (6.*i))`.
fn hdiv_table() -> [i32; 256] {
    let mut t = [0i32; 256];
    for (i, entry) in t.iter_mut().enumerate().skip(1) {
        *entry = ((HRANGE_HSV << HSV_SHIFT) as f64 / (6.0 * i as f64)).round() as i32;
    }
    t
}

/// `sdiv_table` of `cv::RGB2HSV_b`: `saturate_cast<int>((255 << 12) / (1.*i))`.
fn sdiv_table() -> [i32; 256] {
    let mut t = [0i32; 256];
    for (i, entry) in t.iter_mut().enumerate().skip(1) {
        *entry = ((255i32 << HSV_SHIFT) as f64 / i as f64).round() as i32;
    }
    t
}

/// The per-pixel kernel of `cv::RGB2HSV_b`, with the lookup tables passed in.
fn rgb_pixel_to_hsv(
    r: f64,
    g: f64,
    b: f64,
    hdiv: &[i32; 256],
    sdiv: &[i32; 256],
    hsv: &mut [f64; 3],
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

    hsv[0] = h.clamp(0, 255) as f64;
    hsv[1] = s.clamp(0, 255) as f64;
    hsv[2] = v;
}

/// `cv::cvtColor(rgb, hsv, COLOR_RGB2HSV_FULL)` for an interleaved RGB image,
/// with every sample scaled on the way in.
///
/// `rgb` holds `width * height * 3` interleaved samples (any type convertible
/// to `f64`); `scale` converts them to the `0..=255` range the conversion is
/// defined on, which lets a caller that already holds normalised `[0, 1]`
/// samples use them directly. The returned planes hold H (full hue range,
/// 0..=255), S and V.
pub(crate) fn rgb_to_hsv_full<T: Copy + Into<f64> + Sync>(
    rgb: &[T],
    width: usize,
    height: usize,
    scale: f64,
) -> [Vec<f64>; 3] {
    assert_eq!(rgb.len(), width * height * 3, "rgb buffer size mismatch");
    let hdiv = hdiv_table();
    let sdiv = sdiv_table();
    let mut hsv: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0f64; width * height]);

    rgb_pixels_to_hsv(
        |pixel, out| {
            *out = [
                rgb[3 * pixel].into() * scale,
                rgb[3 * pixel + 1].into() * scale,
                rgb[3 * pixel + 2].into() * scale,
            ];
        },
        &hdiv,
        &sdiv,
        &mut hsv,
    );

    hsv
}

/// Drive the RGB->HSV kernel; `read` supplies the source triple of one pixel.
fn rgb_pixels_to_hsv<F>(read: F, hdiv: &[i32; 256], sdiv: &[i32; 256], hsv: &mut [Vec<f64>; 3])
where
    F: Fn(usize, &mut [f64; 3]) + Sync,
{
    // Pixels are independent, and each thread writes a contiguous range of
    // every plane, so the split cannot change a single value.
    parallel_planes(hsv, worker_count(), |planes, start| {
        let [h, s, v] = planes else { unreachable!() };
        let mut src = [0.0f64; 3];
        let mut out = [0.0f64; 3];
        for (local, pixel) in (start..start + h.len()).enumerate() {
            read(pixel, &mut src);
            rgb_pixel_to_hsv(src[0], src[1], src[2], hdiv, sdiv, &mut out);
            h[local] = out[0];
            s[local] = out[1];
            v[local] = out[2];
        }
    });
}

/// `cv::HSV2RGB_native()` for a single pixel; returns (b, g, r) in 0..=1.
#[inline]
fn hsv_pixel_to_rgb(h: f32, s: f32, v: f32, hscale: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (v, v, v);
    }

    const SECTOR_DATA: [[usize; 3]; 6] = [
        [1, 3, 0],
        [1, 0, 2],
        [3, 0, 1],
        [0, 2, 1],
        [0, 1, 3],
        [2, 1, 0],
    ];

    // ComputeSectorAndClampedH()
    let mut h = h * hscale;
    let sector_floor = h.floor(); // cvFloor
    h -= sector_floor;
    let mut sector = sector_floor as i32 % 6;
    if sector < 0 {
        sector += 6;
    }

    let tab = [
        v,
        v * (1.0 - s),
        v * (1.0 - s * h),
        v * (1.0 - s * (1.0 - h)),
    ];

    let d = SECTOR_DATA[sector as usize];
    (tab[d[0]], tab[d[1]], tab[d[2]])
}

/// `cvRound`: round half to even, like OpenCV's `cvRound` (SSE `cvtss2si`).
#[inline]
fn cv_round(x: f32) -> f32 {
    x.round_ties_even()
}

/// `saturate_cast<uchar>` for a float that is rounded with `cvRound` first.
#[inline]
fn saturate_u8(x: f32) -> u8 {
    cv_round(x).clamp(0.0, 255.0) as u8
}

/// Run `f` over every element of `dst`, split into contiguous chunks across
/// threads when the slice is large enough for the hand-off to pay for itself.
///
/// The slice is split into disjoint chunks, so the threads never alias and the
/// per-element work is identical to the serial version.
pub(crate) fn parallel_fill<T, F>(dst: &mut [T], workers: usize, f: F)
where
    T: Send,
    F: Fn(&mut T, usize) + Sync,
{
    /// Below this many elements the thread hand-off costs more than the work.
    const THRESHOLD: usize = 1 << 15;
    let len = dst.len();
    let workers = workers.min(len / THRESHOLD).max(1);
    if workers <= 1 {
        for (index, value) in dst.iter_mut().enumerate() {
            f(value, index);
        }
        return;
    }
    let chunk = len.div_ceil(workers);
    let mut offset = 0usize;
    std::thread::scope(|scope| {
        for slice in dst.chunks_mut(chunk) {
            let start = offset;
            offset += slice.len();
            let f = &f;
            scope.spawn(move || {
                for (index, value) in slice.iter_mut().enumerate() {
                    f(value, start + index);
                }
            });
        }
    });
}

/// [`parallel_fill`] for a kernel that writes several planes at once.
///
/// `out` is split in lockstep, so each thread gets a disjoint index range of
/// every plane and the split cannot change a value.
pub(crate) fn parallel_planes<T, F>(out: &mut [Vec<T>], workers: usize, f: F)
where
    T: Send,
    F: Fn(&mut [&mut [T]], usize) + Sync,
{
    /// Below this many elements the thread hand-off costs more than the work.
    const THRESHOLD: usize = 1 << 15;
    let n = out.first().map(|plane| plane.len()).unwrap_or(0);
    if n == 0 {
        return;
    }
    let workers = workers.min(n / THRESHOLD).max(1);
    if workers <= 1 {
        let mut parts: Vec<&mut [T]> = out.iter_mut().map(|plane| plane.as_mut_slice()).collect();
        f(&mut parts, 0);
        return;
    }

    let chunk = n.div_ceil(workers);
    let mut offset = 0usize;
    std::thread::scope(|scope| {
        let mut rest: Vec<&mut [T]> = out.iter_mut().map(|plane| plane.as_mut_slice()).collect();
        while offset < n {
            let start = offset;
            let len = chunk.min(n - offset);
            let mut parts: Vec<&mut [T]> = Vec::with_capacity(rest.len());
            let mut tail: Vec<&mut [T]> = Vec::with_capacity(rest.len());
            for plane in rest {
                let (head, tail_rest) = plane.split_at_mut(len);
                parts.push(head);
                tail.push(tail_rest);
            }
            rest = tail;
            let f = &f;
            scope.spawn(move || f(&mut parts, start));
            offset += len;
        }
    });
}

/// How many threads to use: one per core, capped at 64.
///
/// `ADAPTIVE_ENHANCE_WORKERS` overrides it (clamped to `1..=64`).
pub(crate) fn worker_count() -> usize {
    if let Ok(value) = std::env::var("ADAPTIVE_ENHANCE_WORKERS") {
        if let Ok(parsed) = value.trim().parse::<usize>() {
            return parsed.clamp(1, 64);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(64)
}

// ---------------------------------------------------------------------------
// Gaussian filtering
// ---------------------------------------------------------------------------

/// `cv::getGaussianKernel(ksize, sigma)` (which ignores `ksize` for `sigma > 0`).
fn gaussian_kernel(ksize: usize, sigma: f64) -> Vec<f64> {
    assert!(ksize > 0 && ksize % 2 == 1, "kernel size must be odd");
    let scale2x = -0.5 / (sigma * sigma);
    let mut kernel = vec![0.0f64; ksize];
    let mut sum = 0.0f64;
    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as f64 - (ksize as f64 - 1.0) * 0.5;
        let t = x * x * scale2x;
        *k = t.exp();
        sum += *k;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }
    kernel
}

/// Vertical convolution of an 8-bit single channel image with a `f64` kernel,
/// using `BORDER_CONSTANT` (zero) padding, like `cv::filter2D`.
///
/// The C++ code hands `cv::getGaussianKernel(ksize, sigma)` to `cv::filter2D`.
/// That helper returns a `ksize x 1` *column* vector, so the kernel is 5x1 and
/// the filtering is a purely vertical convolution (there is no horizontal pass).
fn filter_vertical<T: Copy + Into<f64> + Sync>(
    src: &[T],
    width: usize,
    height: usize,
    kernel: &[f64],
) -> Vec<f64> {
    assert_eq!(src.len(), width * height, "source buffer size mismatch");
    let radius = kernel.len() / 2;
    let mut dst = vec![0.0f64; width * height];

    // One output sample: the kernel is a column vector, so only the row
    // changes, and the neighbourhood is clipped at the image border
    // (`BORDER_CONSTANT`).
    let convolved = |y: usize, x: usize| {
        let mut acc = 0.0f64;
        for (i, k) in kernel.iter().enumerate() {
            let yi = y as isize + i as isize - radius as isize;
            if yi >= 0 && (yi as usize) < height {
                acc += k * src[yi as usize * width + x].into();
            }
        }
        acc
    };

    // Pixels are independent, so the split across threads is exact.
    parallel_fill(&mut dst, worker_count(), |value, index| {
        *value = convolved(index / width, index % width);
    });

    dst
}

/// `(255 + k) * v / (max(v, v_g) + k)`.
///
/// `k` is proportional to `mean(S)`, so on a fully black or fully desaturated
/// image the denominator can be exactly zero; the ratio is then left at zero
/// instead of producing a NaN.
#[inline]
fn ratio(v: f64, v_g: f64, k: f64) -> f64 {
    let denominator = v.max(v_g) + k;
    if denominator == 0.0 {
        0.0
    } else {
        ((255.0 + k) * v) * (1.0 / denominator)
    }
}

/// Weight `w1` of the first principal component of the 2x2 covariance matrix
/// of `(x, y)`, matching the C++ computation
/// `eigenVectors[0][0] / (eigenVectors[0][0] + eigenVectors[0][1])`.
///
/// `cv::eigen` returns the eigenvectors as rows, sorted by descending
/// eigenvalue; for a symmetric 2x2 matrix the principal eigenvector is
/// `(cos t, sin t)` with `t = 0.5 * atan2(2b, a - c)`.
fn principal_weight(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut a = 0.0f64;
    let mut b = 0.0f64;
    let mut c = 0.0f64;
    for (xi, yi) in x.iter().zip(y) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        a += dx * dx;
        b += dx * dy;
        c += dy * dy;
    }

    let theta = 0.5 * (2.0 * b).atan2(a - c);
    let v0 = theta.cos();
    let v1 = theta.sin();

    v0 / (v0 + v1)
}

// ---------------------------------------------------------------------------
// The enhancement itself
// ---------------------------------------------------------------------------

/// Port of `adaptiveImageEnhancement()` for an 8-bit RGB image.
///
/// `rgb` holds `width * height * 3` interleaved bytes and the returned buffer
/// has the same layout. An alpha channel, if the image has one, is not part of
/// this buffer and is therefore untouched, like in the C++ original.
pub fn adaptive_enhance_rgb(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert!(width > 0 && height > 0, "empty image");
    let (planes, _) = adaptive_enhance_interleaved(rgb, width, height, 1.0);
    let mut out = vec![0u8; width * height * 3];
    parallel_fill(&mut out, worker_count(), |value, index| {
        *value = planes[index % 3][index / 3];
    });
    out
}

/// [`adaptive_enhance_rgb`] on an interleaved image of any sample type, with a
/// scale factor applied as the samples are read.
///
/// `scale` converts the samples to the `0..=255` range the algorithm is defined
/// on, so a caller holding normalised samples can use them directly. The
/// returned value channel is `F`, the enhanced value channel the fusion uses as
/// its illumination estimate.
pub fn adaptive_enhance_interleaved<T: Copy + Into<f64> + Sync>(
    rgb: &[T],
    width: usize,
    height: usize,
    scale: f64,
) -> ([Vec<u8>; 3], Vec<f64>) {
    let hsv = rgb_to_hsv_full(rgb, width, height, scale);
    adaptive_enhance_hsv(&hsv, width, height)
}

/// The enhancement itself, given the HSV planes of the input: three vertical
/// blurs, the two ratios, and the principal-component weighting of them.
fn adaptive_enhance_hsv(
    hsv: &[Vec<f64>; 3],
    width: usize,
    height: usize,
) -> ([Vec<u8>; 3], Vec<f64>) {
    let n = width * height;
    assert!(width > 0 && height > 0, "empty image");
    assert_eq!(hsv[0].len(), n, "hsv buffer size mismatch");

    let s_channel = &hsv[1];
    let v_channel = &hsv[2];

    // Three vertical Gaussian blurs of V, averaged.
    let mut v_g = vec![0.0f64; n];
    for sigma in SIGMAS {
        let kernel = gaussian_kernel(KSIZE, sigma);
        let blurred = filter_vertical(v_channel, width, height, &kernel);
        parallel_fill(&mut v_g, worker_count(), |acc, index| {
            *acc += blurred[index] / 3.0;
        });
    }

    // cv::mean(S)
    let avg_s = s_channel.iter().sum::<f64>() / n as f64;
    let k1 = 0.1 * avg_s;
    let k2 = avg_s;

    let mut v1 = vec![0.0f64; n];
    let mut v2 = vec![0.0f64; n];
    // Both ratios are per-pixel, as is the average of the blurs, so the loops
    // are split across threads without changing a value.
    parallel_fill(&mut v1, worker_count(), |value, i| {
        *value = ratio(v_channel[i], v_g[i], k1);
    });
    parallel_fill(&mut v2, worker_count(), |value, i| {
        *value = ratio(v_channel[i], v_g[i], k2);
    });

    // Principal component of (V1, V2) -> weights w1, w2.
    let w1 = principal_weight(&v1, &v2);
    let w2 = 1.0 - w1;

    // F is written over V1, which nothing reads afterwards.
    let v2_ref = &v2;
    parallel_fill(&mut v1, worker_count(), |value, i| {
        *value = w1 * *value + w2 * v2_ref[i];
    });
    let f_channel = v1;

    // HSV -> RGB straight into the output bytes: the kernel rounds once, so
    // nothing is lost by writing its result directly.
    let mut out: [Vec<u8>; 3] = std::array::from_fn(|_| vec![0u8; n]);
    // The conversion is per pixel, and every thread writes its own range of the
    // output planes, so the split is exact.
    let h0 = &hsv[0];
    let h1 = &hsv[1];
    let f_channel_ref = &f_channel;
    parallel_planes(&mut out, worker_count(), |planes, start| {
        let [r_plane, g_plane, b_plane]: &mut [&mut [u8]; 3] = planes.try_into().unwrap();
        for (local, ((r_out, g_out), b_out)) in r_plane
            .iter_mut()
            .zip(g_plane.iter_mut())
            .zip(b_plane.iter_mut())
            .enumerate()
        {
            let i = start + local;
            // `out_hsv[3 * i + 2] = f.round_ties_even().clamp(0.0, 255.0) as u8`:
            // F is quantised to the 0..=255 sample range *before* the HSV->RGB
            // kernel sees it, exactly as `cv::cvtColor` does.
            let f = f_channel_ref[i].round_ties_even().clamp(0.0, 255.0);
            let (b, g, r) = hsv_pixel_to_rgb(
                h0[i] as f32,
                h1[i] as f32 * (1.0f32 / 255.0),
                f as f32 * (1.0f32 / 255.0),
                HSCALE,
            );
            // `saturate_u8(r * 255.0)` for each output channel, in the RGB
            // order the kernel returns them.
            *r_out = saturate_u8(r * 255.0);
            *g_out = saturate_u8(g * 255.0);
            *b_out = saturate_u8(b * 255.0);
        }
    });

    (out, f_channel)
}
