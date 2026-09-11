# adaptive-enhance

[![CI](https://github.com/jacobsparts/adaptive-enhance/actions/workflows/ci.yml/badge.svg)](https://github.com/jacobsparts/adaptive-enhance/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A small, standalone Rust implementation of OpenCE's adaptive image enhancement
algorithm. It brightens shadows and increases local contrast in low-light
images without requiring OpenCV.

The project provides both:

- a Unix-friendly command that reads PNG data from standard input and writes
  PNG data to standard output; and
- a Rust library for enhancing decoded RGB buffers or in-memory PNG data.

## Why this exists

Adaptive enhancement can reveal product contours that disappear into a dark or
flatly lit surface. That can make an otherwise unusable image detectable by a
vision model.

It is **not** a universally beneficial preprocessing step. On an already
well-lit product photo, enhancement may amplify texture across the entire table
or backdrop. A segmentation or localization model can then mistake much of the
scene for foreground.

For that reason, our intended product-photo pipeline uses this as a fallback:

1. run the original image through the detector;
2. if the detector returns no usable detection, enhance the original image;
3. retry detection on the enhanced image; and
4. retain the original image for display and downstream work unless the
   enhanced result is specifically needed.

This repository is model-agnostic. We use the strategy with
[LocateAnything](https://research.nvidia.com/labs/lpr/locate-anything/), but the
project is not affiliated with or endorsed by NVIDIA.

## Quick start

A stable Rust toolchain (Rust 1.85 or newer) is required.

```console
git clone https://github.com/jacobsparts/adaptive-enhance.git
cd adaptive-enhance
cargo build --release
./target/release/adaptive-enhance < input.png > enhanced.png
```

The command is deliberately pipe-oriented, so it also composes with other
programs:

```console
cat input.png | adaptive-enhance > enhanced.png
```

Errors are written to stderr and result in a non-zero exit status. No partial
PNG is written when processing fails.

### Supported PNG input

- grayscale and grayscale + alpha
- RGB and RGBA
- palette images
- 8-bit and 16-bit samples

Output is always 8-bit RGB or RGBA. Alpha is preserved unchanged. For 16-bit
input, each sample is reduced to its most significant byte before enhancement.

## Library use

Enhance an interleaved 8-bit RGB buffer:

```rust
let enhanced = adaptive_enhance::adaptive_enhance_rgb(&rgb, width, height);
```

The input must contain exactly `width * height * 3` bytes in RGB order. The
returned buffer has the same layout and dimensions.

Or process a complete PNG in memory:

```rust
let output_png = adaptive_enhance::png_io::enhance_png(&input_png)?;
```

The PNG helpers return `Result<_, String>`. The lower-level RGB function asserts
that its dimensions and buffer length are valid.

## Algorithm

The implementation ports `adaptiveImageEnhancement()` from the
[OpenCE project](https://baidut.github.io/OpenCE/caip2017.html):

1. convert RGB to full-range HSV;
2. vertically filter the value channel with three 5x1 Gaussian kernels
   (`sigma = 15, 80, 250`) and average the results;
3. derive two candidate value channels using the mean saturation;
4. blend those candidates using the principal eigenvector of their 2x2
   covariance matrix; and
5. replace the value channel and convert back to RGB.

Only luminance/value is adapted; hue and saturation are retained through the
HSV round trip.

### OpenCV compatibility

The original implementation relies on details of OpenCV that differ from naive
textbook formulas. This port reproduces those details without linking OpenCV,
including:

- OpenCV's fixed-point 8-bit `RGB2HSV_FULL` path;
- its floating-point `HSV2RGB_FULL` path and round-half-to-even behavior;
- the original vertical-only Gaussian filtering with zero padding;
- unnormalized covariance sums; and
- round-half-to-even conversion of the final value channel.

Tests compare conversion kernels and complete PNG outputs against reference
values generated with OpenCV 4.10. In rare HSV-to-RGB half-way cases, OpenCV's
internal `float32` rounding can differ by one 8-bit step. The known cases are
documented in `tests/opencv_compat.rs`.

## Validation and development

The test suite has no Python or OpenCV runtime dependency:

```console
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

The checked-in golden fixtures were generated with OpenCV 4.10
(`opencv-python-headless==4.10.0.84`). CI runs formatting, linting, and tests on
every push and pull request.

## Repository layout

```text
src/lib.rs             enhancement algorithm and color conversions
src/png_io.rs          PNG decoding, encoding, and in-memory pipeline
src/main.rs            stdin-to-stdout command-line interface
tests/opencv_compat.rs OpenCV compatibility checks
tests/pipeline.rs      end-to-end CLI and fixture tests
tests/fixtures/        source images and OpenCV reference outputs
```

## Limitations

- PNG is the only encoded image format supported by the CLI.
- The algorithm is intentionally fixed to the original parameters; there are
  no strength or threshold controls.
- Enhancement is global and may make background texture more salient. Evaluate
  the output against your own detector and dataset rather than applying it
  unconditionally.
- EXIF and other source-image metadata are not preserved when the PNG is
  re-encoded.

## Attribution and license

This is an independent Rust port of work from
[OpenCE](https://github.com/baidut/OpenCE), originally copyright Zhenqiang Ying
and released under the MIT License. See [NOTICE](NOTICE) for attribution.

This repository is also released under the [MIT License](LICENSE).
