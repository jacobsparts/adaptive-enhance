//! Golden-value tests against the OpenCV behaviour the C++ original relies on.
//!
//! The expected values were produced with OpenCV 4.10
//! (`cv::cvtColor(..., COLOR_RGB2HSV_FULL)`, `COLOR_HSV2RGB_FULL` and
//! `cv::getGaussianKernel`) and are reproduced here so that the port can be
//! checked without an OpenCV installation.

use adaptive_enhance::{gaussian_kernel, hsv_to_rgb_full, rgb_to_hsv_full, SIGMAS};

/// `(rgb, hsv)` pairs from `cv::cvtColor(rgb, COLOR_RGB2HSV_FULL)`.
const RGB_TO_HSV_FULL: [([u8; 3], [u8; 3]); 16] = [
    ([0, 0, 0], [0, 0, 0]),
    ([255, 255, 255], [0, 0, 255]),
    ([255, 0, 0], [0, 255, 255]),
    ([0, 255, 0], [85, 255, 255]),
    ([0, 0, 255], [171, 255, 255]),
    ([255, 255, 0], [43, 255, 255]),
    ([0, 255, 255], [128, 255, 255]),
    ([255, 0, 255], [213, 255, 255]),
    ([128, 128, 128], [0, 0, 128]),
    ([12, 34, 56], [149, 200, 56]),
    ([200, 100, 50], [14, 191, 200]),
    ([1, 2, 3], [149, 170, 3]),
    ([250, 3, 7], [255, 252, 250]),
    ([7, 250, 3], [85, 252, 250]),
    ([3, 7, 250], [170, 252, 250]),
    ([90, 140, 200], [151, 140, 200]),
];

/// `(hsv, rgb)` pairs from `cv::cvtColor(hsv, COLOR_HSV2RGB_FULL)`.
const HSV_TO_RGB_FULL: [([u8; 3], [u8; 3]); 15] = [
    ([0, 255, 255], [255, 0, 0]),
    ([85, 255, 255], [0, 255, 0]),
    ([170, 255, 255], [0, 0, 255]),
    ([43, 255, 255], [252, 255, 0]),
    ([255, 255, 255], [255, 0, 0]),
    ([0, 0, 0], [0, 0, 0]),
    ([0, 0, 128], [128, 128, 128]),
    ([100, 0, 200], [200, 200, 200]),
    ([249, 242, 219], [219, 11, 41]),
    ([17, 64, 200], [200, 170, 150]),
    ([200, 17, 64], [63, 60, 64]),
    ([64, 200, 17], [10, 17, 4]),
    ([128, 128, 128], [64, 127, 128]),
    ([200, 200, 200], [154, 43, 200]),
    ([30, 90, 240], [240, 215, 155]),
];

/// `cv::getGaussianKernel(5, sigma)` for the three sigmas used by the original.
const GAUSSIAN_KERNELS: [[f64; 5]; 3] = [
    [
        0.19911170735683686,
        0.2004435532929103,
        0.2008894787005057,
        0.2004435532929103,
        0.19911170735683686,
    ],
    [
        0.1999687507325808,
        0.20001562390126865,
        0.20003125073230105,
        0.20001562390126865,
        0.1999687507325808,
    ],
    [
        0.19999680000768016,
        0.20000159998847988,
        0.20000320000767985,
        0.20000159998847988,
        0.19999680000768016,
    ],
];

#[test]
fn rgb_to_hsv_full_matches_opencv() {
    for (rgb, expected) in RGB_TO_HSV_FULL {
        let pixels: Vec<u8> = rgb.to_vec();
        let hsv = rgb_to_hsv_full(&pixels, 1, 1);
        assert_eq!(
            hsv.as_slice(),
            expected.as_slice(),
            "rgb {rgb:?} should convert to {expected:?}"
        );
    }
}

#[test]
fn hsv_to_rgb_full_matches_opencv() {
    for (hsv, expected) in HSV_TO_RGB_FULL {
        let pixels: Vec<u8> = hsv.to_vec();
        let rgb = hsv_to_rgb_full(&pixels, 1, 1);
        assert_eq!(
            rgb.as_slice(),
            expected.as_slice(),
            "hsv {hsv:?} should convert to {expected:?}"
        );
    }
}

/// HSV triples for which OpenCV's own float32 arithmetic lands exactly on a
/// `x.5` tie of the final `cvRound`, so the last bit is decided by rounding
/// noise inside `cv::HSV2RGB_b` rather than by the algorithm. The port is
/// allowed to be off by one 8-bit step for these (measured rate: 5 pixels out
/// of 1.5 million random triples, always 1 LSB).
const HSV_TO_RGB_FULL_TIES: [([u8; 3], [u8; 3]); 1] = [([154, 209, 181], [33, 88, 181])];

#[test]
fn hsv_to_rgb_full_tie_cases_are_within_one_step() {
    for (hsv, expected) in HSV_TO_RGB_FULL_TIES {
        let got = hsv_to_rgb_full(&hsv, 1, 1);
        for (channel, (got, want)) in got.iter().zip(expected).enumerate() {
            let delta = (*got as i32 - want as i32).abs();
            assert!(
                delta <= 1,
                "hsv {hsv:?} channel {channel}: got {got}, OpenCV gives {want}"
            );
        }
    }
}

#[test]
fn gaussian_kernels_match_opencv() {
    for (sigma, expected) in SIGMAS.iter().zip(GAUSSIAN_KERNELS) {
        let kernel = gaussian_kernel(5, *sigma);
        for (got, want) in kernel.iter().zip(expected) {
            assert!(
                (got - want).abs() <= 1e-15,
                "sigma {sigma}: kernel {kernel:?} != {expected:?}"
            );
        }
        // cv::getGaussianKernel normalises the kernel to sum to one.
        assert!((kernel.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }
}

/// The achromatic case (`S == 0`) is special-cased by `cv::HSV2RGB_native`.
#[test]
fn grey_stays_grey() {
    let hsv = [17u8, 0, 200];
    assert_eq!(hsv_to_rgb_full(&hsv, 1, 1), vec![200, 200, 200]);
}
