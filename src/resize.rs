//! Two resampling kernels used by the exposure fusion pipeline.
//!
//! `area_resize` reproduces `torch.nn.functional.interpolate(mode="area")` and
//! `bicubic_resize_f32` reproduces `mode="bicubic", align_corners=False`, which
//! is what the reference PyTorch implementation (the PyTorch reference) uses to
//! downscale the illumination map and the `isBad` mask. Both operate on a
//! single channel stored row-major, `width * height` samples.

/// torch's `area_pixel_compute_scale`: the scale used for input/output
/// coordinate mapping in `mode="area"`.
fn area_scale(input_size: usize, output_size: usize) -> f64 {
    if output_size > 1 {
        input_size as f64 / output_size as f64
    } else {
        0.0
    }
}

/// torch's `area_pixel_compute_source_index` (with `cubic=false`); the
/// bicubic path and the area path agree on this coordinate mapping.
fn area_source_index(scale: f64, dst_index: usize, align_corners: bool) -> f64 {
    if align_corners {
        scale * dst_index as f64
    } else {
        scale * (dst_index as f64 + 0.5) - 0.5
    }
}

/// `torch.nn.functional.interpolate(x, mode="area")` for one channel.
///
/// Each output sample is the mean of the input samples falling in its
/// footprint, exactly as `adaptive_avg_pool2d` computes it. That is identical
/// for both `align_corners` settings, so no flag is needed.
pub fn area_resize(
    src: &[f64],
    width: usize,
    height: usize,
    out_w: usize,
    out_h: usize,
) -> Vec<f64> {
    assert_eq!(src.len(), width * height, "source buffer size mismatch");
    assert!(out_w > 0 && out_h > 0, "empty output");
    let mut out = vec![0.0f64; out_w * out_h];

    let height_scale = area_scale(height, out_h);
    let width_scale = area_scale(width, out_w);

    for oy in 0..out_h {
        // adaptive_avg_pool2d: start = floor(oy * ih / oh), end = ceil((oy+1) * ih / oh),
        // both clamped to the input, with at least one sample in the window.
        let start_h =
            ((oy as f64 * height_scale).floor() as isize).clamp(0, height as isize - 1) as usize;
        let end_h = ((((oy + 1) as f64 * height_scale).ceil() as isize).max(start_h as isize + 1))
            .clamp(start_h as isize + 1, height as isize) as usize;
        let rows = end_h - start_h;

        for ox in 0..out_w {
            let start_w =
                ((ox as f64 * width_scale).floor() as isize).clamp(0, width as isize - 1) as usize;
            let end_w = ((((ox + 1) as f64 * width_scale).ceil() as isize)
                .max(start_w as isize + 1))
            .clamp(start_w as isize + 1, width as isize) as usize;
            let cols = end_w - start_w;

            let mut acc = 0.0f64;
            for y in start_h..end_h {
                let row = &src[y * width..y * width + width];
                acc += row[start_w..end_w].iter().sum::<f64>();
            }
            out[oy * out_w + ox] = acc / (rows * cols) as f64;
        }
    }

    out
}

/// [`area_resize`] on an interleaved RGB image, writing interleaved RGB.
///
/// The accumulation is the same one [`area_resize`] performs on one channel -
/// row sums added in row order, then divided by the footprint area - so the
/// result is identical to running [`area_resize`] on each channel of a split
/// image, without materialising the split image (three full-image `f64` planes,
/// 577 MB on a 24-megapixel frame, for a 50x50 result).
pub fn area_resize_rgb(
    src: &[f64],
    width: usize,
    height: usize,
    out_w: usize,
    out_h: usize,
) -> Vec<f64> {
    assert_eq!(src.len(), width * height * 3, "source buffer size mismatch");
    assert!(out_w > 0 && out_h > 0, "empty output");
    let mut out = vec![0.0f64; out_w * out_h * 3];

    let height_scale = area_scale(height, out_h);
    let width_scale = area_scale(width, out_w);

    for oy in 0..out_h {
        let start_h =
            ((oy as f64 * height_scale).floor() as isize).clamp(0, height as isize - 1) as usize;
        let end_h = ((((oy + 1) as f64 * height_scale).ceil() as isize).max(start_h as isize + 1))
            .clamp(start_h as isize + 1, height as isize) as usize;
        let rows = end_h - start_h;

        for ox in 0..out_w {
            let start_w =
                ((ox as f64 * width_scale).floor() as isize).clamp(0, width as isize - 1) as usize;
            let end_w = ((((ox + 1) as f64 * width_scale).ceil() as isize)
                .max(start_w as isize + 1))
            .clamp(start_w as isize + 1, width as isize) as usize;
            let cols = end_w - start_w;

            let mut acc = [0.0f64; 3];
            for y in start_h..end_h {
                let row = &src[y * width * 3..(y + 1) * width * 3];
                for channel in 0..3 {
                    acc[channel] += row[start_w * 3 + channel..end_w * 3]
                        .iter()
                        .step_by(3)
                        .sum::<f64>();
                }
            }
            let area = (rows * cols) as f64;
            for (channel, value) in acc.iter().enumerate() {
                out[(oy * out_w + ox) * 3 + channel] = value / area;
            }
        }
    }

    out
}

/// torch's `get_cubic_upsample_coefficients(t)`.
fn cubic_upsample_coefficients(t: f64) -> [f64; 4] {
    let a = -0.75f64;
    [
        a * (t + 1.0).powi(3) - 5.0 * a * (t + 1.0).powi(2) + 8.0 * a * (t + 1.0) - 4.0 * a,
        (a + 2.0) * t.powi(3) - (a + 3.0) * t.powi(2) + 1.0,
        (a + 2.0) * (1.0 - t).powi(3) - (a + 3.0) * (1.0 - t).powi(2) + 1.0,
        a * (2.0 - t).powi(3) - 5.0 * a * (2.0 - t).powi(2) + 8.0 * a * (2.0 - t) - 4.0 * a,
    ]
}

/// torch's `get_cubic_coefficients(t1, t2)` as a 4x4 matrix.
fn cubic_coefficients(t1: f64, t2: f64) -> [[f64; 4]; 4] {
    let c1 = cubic_upsample_coefficients(t1);
    let c2 = cubic_upsample_coefficients(t2);
    let mut coeffs = [[0.0f64; 4]; 4];
    for (i, coeff) in coeffs.iter_mut().enumerate() {
        for (j, value) in coeff.iter_mut().enumerate() {
            *value = c1[i] * c2[j];
        }
    }
    coeffs
}

/// `torch.nn.functional.interpolate(x, mode="bicubic", align_corners=False)`.
///
/// The input is used as `f32` (torch's default float type) and the separable
/// convolution is exactly `(A_m * in) * A_m^T` of the two 1-D cubic
/// interpolations, like `upsample_bicubic2d_out_frame` computes it.
pub fn bicubic_resize_f32(
    src: &[f32],
    width: usize,
    height: usize,
    out_w: usize,
    out_h: usize,
) -> Vec<f32> {
    assert_eq!(src.len(), width * height, "source buffer size mismatch");
    assert!(out_w > 0 && out_h > 0, "empty output");
    let mut out = vec![0.0f32; out_w * out_h];

    let height_scale = area_scale(height, out_h);
    let width_scale = area_scale(width, out_w);

    for oy in 0..out_h {
        let real_y = area_source_index(height_scale, oy, false).max(0.0);
        let in_y = real_y.floor();
        let t_y = real_y - in_y;
        let y0 = in_y as usize;

        // The outermost samples are clamped, the two inner ones may fall
        // outside of the image (torch lets the accessor handle that).
        let y_indices = [
            (y0 as isize - 1).clamp(0, height as isize) as usize,
            y0.min(height - 1),
            (y0 + 1).min(height - 1),
            (y0 + 2).min(height - 1),
        ];

        for ox in 0..out_w {
            let real_x = area_source_index(width_scale, ox, false).max(0.0);
            let in_x = real_x.floor();
            let t_x = real_x - in_x;
            let x0 = in_x as usize;

            let x_indices = [
                (x0 as isize - 1).clamp(0, width as isize) as usize,
                x0.min(width - 1),
                (x0 + 1).min(width - 1),
                (x0 + 2).min(width - 1),
            ];

            let coeffs = cubic_coefficients(t_x, t_y);

            let mut acc = 0.0f32;
            for (i, &iy) in y_indices.iter().enumerate() {
                for (j, &ix) in x_indices.iter().enumerate() {
                    acc += coeffs[i][j] as f32 * src[iy * width + ix];
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }

    out
}

/// [`bicubic_resize_f32`] on a `bool` source, read as `0.0` / `1.0`.
///
/// Same accumulation, same `f32` weights; the point is that the caller does not
/// have to expand the mask to a full-image `f32` plane (96 MB on a
/// 24-megapixel frame) only to resample it to 50x50.
pub fn bicubic_resize_bool(
    src: &[bool],
    width: usize,
    height: usize,
    out_w: usize,
    out_h: usize,
) -> Vec<f32> {
    assert_eq!(src.len(), width * height, "source buffer size mismatch");
    assert!(out_w > 0 && out_h > 0, "empty output");
    let mut out = vec![0.0f32; out_w * out_h];

    let height_scale = area_scale(height, out_h);
    let width_scale = area_scale(width, out_w);

    for oy in 0..out_h {
        let real_y = area_source_index(height_scale, oy, false).max(0.0);
        let in_y = real_y.floor();
        let t_y = real_y - in_y;
        let y0 = in_y as usize;

        let y_indices = [
            (y0 as isize - 1).clamp(0, height as isize) as usize,
            y0.min(height - 1),
            (y0 + 1).min(height - 1),
            (y0 + 2).min(height - 1),
        ];

        for ox in 0..out_w {
            let real_x = area_source_index(width_scale, ox, false).max(0.0);
            let in_x = real_x.floor();
            let t_x = real_x - in_x;
            let x0 = in_x as usize;

            let x_indices = [
                (x0 as isize - 1).clamp(0, width as isize) as usize,
                x0.min(width - 1),
                (x0 + 1).min(width - 1),
                (x0 + 2).min(width - 1),
            ];

            let coeffs = cubic_coefficients(t_x, t_y);

            let mut acc = 0.0f32;
            for (i, &iy) in y_indices.iter().enumerate() {
                for (j, &ix) in x_indices.iter().enumerate() {
                    acc += coeffs[i][j] as f32 * (src[iy * width + ix] as u8 as f32);
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }

    out
}
