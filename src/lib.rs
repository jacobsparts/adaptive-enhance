//! Adaptive image enhancement.
//!
//! A standalone Rust port of the C++ function `adaptiveImageEnhancement()`
//! (see `image_enhancement.cpp`), which is built on OpenCV:
//!
//! ```text
//! BGR -> HSV_FULL, split
//! V_g  = mean of three vertical 5x1 Gaussian blurs of V (sigma 15, 80, 250)
//! V1   = (255 + k1) * V / (max(V, V_g) + k1)     k1 = 0.1 * mean(S)
//! V2   = (255 + k2) * V / (max(V, V_g) + k2)     k2 =      mean(S)
//! F    = w1 * V1 + w2 * V2   (w from the principal eigenvector of the 2x2
//!                             covariance of (V1, V2) over all pixels)
//! HSV_FULL -> BGR with V replaced by round(F)
//! ```
//!
//! Everything is reimplemented here from scratch (no OpenCV dependency), using
//! the same arithmetic OpenCV uses so that results match the original:
//!
//! * the 8-bit `BGR2HSV_FULL` conversion uses OpenCV's fixed-point integer
//!   arithmetic (`hsv_shift = 12` lookup tables, full hue range 0..=255);
//! * the 8-bit `HSV2BGR_FULL` conversion uses OpenCV's float path with
//!   `hscale = 6.0f / 255.0f` (full-range hue) and `cvRound`
//!   (round-half-to-even) when converting back to 8-bit;
//! * the Gaussian filtering, the per-pixel ratios and the covariance are done
//!   in `f64`, exactly like the `CV_64F` matrices of the C++ code.
//!
//! The two 8-bit HSV conversions are ports of the OpenCV kernels and reproduce
//! `cv::cvtColor` (see `tests/opencv_compat.rs`). The only numerical freedom
//! left is the eigenvector of the 2x2 covariance matrix, which is computed in
//! closed form instead of by `cv::eigen`: `w1` is eigenvector-sign invariant
//! and the closed form agrees with `cv::eigen` to ~1e-11 in `f64`, i.e. far
//! below the 1/255 quantization step of `F`.
//!
//! One caveat: for HSV triples whose final `cvRound` lands exactly on a `x.5`
//! tie, OpenCV's own `float32` arithmetic decides the last bit, so such pixels
//! can differ from `cv::cvtColor` by one 8-bit step. Measured rate: 5 pixels
//! out of 1.5 million random triples. See `tests/opencv_compat.rs`.

pub mod png_io;

/// Kernel size of the three Gaussian blurs (C++: `int ksize = 5;`).
pub const KSIZE: usize = 5;
/// Standard deviations of the three Gaussian blurs (C++: 15, 80, 250).
pub const SIGMAS: [f64; 3] = [15.0, 80.0, 250.0];

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

/// `cv::cvtColor(rgb, hsv, COLOR_RGB2HSV_FULL)` for an 8-bit RGB image.
///
/// `rgb` holds `width * height * 3` interleaved bytes; the returned buffer has
/// the same layout and holds H, S, V (H in 0..=255 for the full hue range).
pub fn rgb_to_hsv_full(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert_eq!(rgb.len(), width * height * 3, "rgb buffer size mismatch");
    let hdiv = hdiv_table();
    let sdiv = sdiv_table();
    let mut hsv = vec![0u8; width * height * 3];

    for (src, dst) in rgb.chunks_exact(3).zip(hsv.chunks_exact_mut(3)) {
        // Note: OpenCV's RGB2HSV_b works on (b, g, r) with blueIdx = 2 for the
        // BGR2HSV entry point, i.e. on (r, g, b) for the RGB entry point.
        let r = src[0] as i32;
        let g = src[1] as i32;
        let b = src[2] as i32;

        let v = r.max(g).max(b);
        let vmin = r.min(g).min(b);

        let diff = (v - vmin) as u8 as usize;
        let vr = if v == r { -1i32 } else { 0 };
        let vg = if v == g { -1i32 } else { 0 };

        let s = (diff as i32 * sdiv[v as usize] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
        let mut h = (vr & (g - b))
            + (!vr & ((vg & (b - r + 2 * diff as i32)) + (!vg & (r - g + 4 * diff as i32))));
        h = (h * hdiv[diff] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
        h += if h < 0 { HRANGE_HSV } else { 0 };

        dst[0] = h.clamp(0, 255) as u8;
        dst[1] = s.clamp(0, 255) as u8;
        dst[2] = v as u8;
    }

    hsv
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

/// `cv::cvtColor(hsv, rgb, COLOR_HSV2RGB_FULL)` for an 8-bit image.
///
/// `hsv` holds `width * height * 3` interleaved bytes (H, S, V) and the result
/// is an interleaved 8-bit RGB image.
pub fn hsv_to_rgb_full(hsv: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert_eq!(hsv.len(), width * height * 3, "hsv buffer size mismatch");
    let mut rgb = vec![0u8; width * height * 3];

    for (src, dst) in hsv.chunks_exact(3).zip(rgb.chunks_exact_mut(3)) {
        let h = src[0] as f32;
        let s = src[1] as f32 * (1.0f32 / 255.0);
        let v = src[2] as f32 * (1.0f32 / 255.0);

        let (b, g, r) = hsv_pixel_to_rgb(h, s, v, HSCALE);

        // dst[blueIdx] = saturate_cast<uchar>(b * 255.0f); blueIdx == 2 here.
        dst[0] = saturate_u8(r * 255.0);
        dst[1] = saturate_u8(g * 255.0);
        dst[2] = saturate_u8(b * 255.0);
    }

    rgb
}

// ---------------------------------------------------------------------------
// Gaussian filtering
// ---------------------------------------------------------------------------

/// `cv::getGaussianKernel(ksize, sigma)` (which ignores `ksize` for `sigma > 0`).
pub fn gaussian_kernel(ksize: usize, sigma: f64) -> Vec<f64> {
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
pub fn filter_vertical(src: &[u8], width: usize, height: usize, kernel: &[f64]) -> Vec<f64> {
    let radius = kernel.len() / 2;
    let mut dst = vec![0.0f64; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f64;
            for (i, k) in kernel.iter().enumerate() {
                let yi = y as isize + i as isize - radius as isize;
                if yi >= 0 && (yi as usize) < height {
                    acc += k * src[yi as usize * width + x] as f64;
                }
            }
            dst[y * width + x] = acc;
        }
    }

    dst
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
    assert_eq!(rgb.len(), width * height * 3, "rgb buffer size mismatch");
    assert!(width > 0 && height > 0, "empty image");
    let n = width * height;

    // cv::cvtColor(src, HSV, COLOR_BGR2HSV_FULL); cv::split(HSV, ...);
    // H is only read back when the HSV image is merged again, so the split
    // channels we actually need are S and V.
    let hsv = rgb_to_hsv_full(rgb, width, height);
    let mut s_channel = vec![0u8; n];
    let mut v_channel = vec![0u8; n];
    for i in 0..n {
        s_channel[i] = hsv[3 * i + 1];
        v_channel[i] = hsv[3 * i + 2];
    }

    // Three vertical Gaussian blurs of V, averaged.
    let mut v_g = vec![0.0f64; n];
    for sigma in SIGMAS {
        let kernel = gaussian_kernel(KSIZE, sigma);
        let blurred = filter_vertical(&v_channel, width, height, &kernel);
        for (acc, b) in v_g.iter_mut().zip(&blurred) {
            *acc += b / 3.0;
        }
    }

    // cv::mean(S)
    let avg_s = s_channel.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let k1 = 0.1 * avg_s;
    let k2 = avg_s;

    let mut v1 = vec![0.0f64; n];
    let mut v2 = vec![0.0f64; n];
    for i in 0..n {
        let v = v_channel[i] as f64;
        let denominator1 = v.max(v_g[i]) + k1;
        let denominator2 = v.max(v_g[i]) + k2;
        v1[i] = ((255.0 + k1) * v) * (1.0 / denominator1);
        v2[i] = ((255.0 + k2) * v) * (1.0 / denominator2);
    }

    // Principal component of (V1, V2) -> weights w1, w2.
    let w1 = principal_weight(&v1, &v2);
    let w2 = 1.0 - w1;

    let mut out_hsv = hsv;
    for i in 0..n {
        let f = w1 * v1[i] + w2 * v2[i];
        // F.convertTo(F, CV_8U) on a CV_64F matrix: cvRound (round half to
        // even) followed by saturate_cast<uchar>. Rounding the f64 value
        // directly matters: going through f32 first can round a value like
        // 12.5 up to 12.5000001 and then away from the even neighbour.
        out_hsv[3 * i + 2] = f.round_ties_even().clamp(0.0, 255.0) as u8;
    }

    hsv_to_rgb_full(&out_hsv, width, height)
}
