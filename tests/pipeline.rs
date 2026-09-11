//! End-to-end tests: run the `adaptive-enhance` binary on PNG fixtures and
//! compare the result with the output of the original C++ function
//! (`adaptiveImageEnhancement()`), produced with OpenCV 4.10.

use std::io::Write;
use std::process::{Command, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_adaptive-enhance");

/// Decoded 8-bit image with an alpha plane.
struct Rgba {
    width: usize,
    height: usize,
    /// `width * height * 4` bytes.
    data: Vec<u8>,
}

/// Decode a PNG into RGBA, expanding greyscale and palette images.
fn decode_rgba(bytes: &[u8]) -> Rgba {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().expect("valid PNG header");
    let mut buffer = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer).expect("valid PNG data");
    buffer.truncate(info.buffer_size());

    let width = info.width as usize;
    let height = info.height as usize;
    let samples = match info.bit_depth {
        png::BitDepth::Eight => 1,
        png::BitDepth::Sixteen => 2,
        other => panic!("unexpected bit depth {other:?}"),
    };
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => panic!("indexed image was not expanded"),
    };

    let mut data = Vec::with_capacity(width * height * 4);
    for pixel in buffer.chunks_exact(channels * samples) {
        let sample = |i: usize| pixel[i * samples];
        match channels {
            1 => {
                let g = sample(0);
                data.extend_from_slice(&[g, g, g, 255]);
            }
            2 => {
                let g = sample(0);
                data.extend_from_slice(&[g, g, g, sample(1)]);
            }
            3 => {
                data.extend_from_slice(&[sample(0), sample(1), sample(2), 255]);
            }
            4 => {
                data.extend_from_slice(&[sample(0), sample(1), sample(2), sample(3)]);
            }
            _ => unreachable!(),
        }
    }

    Rgba {
        width,
        height,
        data,
    }
}

/// Run the binary with `input` on stdin and return its stdout.
fn run(input: &[u8]) -> Vec<u8> {
    let mut child = Command::new(BIN)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to run adaptive-enhance");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input)
        .expect("failed to write stdin");
    let output = child.wait_with_output().expect("failed to read stdout");
    assert!(
        output.status.success(),
        "adaptive-enhance failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// Compare the enhanced output of `input` with `expected`, both PNG files.
fn assert_matches_reference(input: &str, expected: &str) {
    let input_bytes = std::fs::read(format!("tests/fixtures/{input}")).unwrap();
    let expected_bytes = std::fs::read(format!("tests/fixtures/{expected}")).unwrap();

    let got = decode_rgba(&run(&input_bytes));
    let want = decode_rgba(&expected_bytes);

    assert_eq!((got.width, got.height), (want.width, want.height));

    let mut differing = 0usize;
    let mut worst = 0i32;
    for (index, (g, w)) in got.data.iter().zip(&want.data).enumerate() {
        let delta = (*g as i32 - *w as i32).abs();
        if delta != 0 {
            differing += 1;
            worst = worst.max(delta);
            assert!(
                delta <= 1,
                "channel {index} (pixel {}) differs by {delta}: got {g}, OpenCV gives {w}",
                index / 4
            );
        }
    }

    // Everything must match exactly; a handful of channels are allowed to be
    // off by one 8-bit step, see tests/opencv_compat.rs for why.
    let pixels = got.width * got.height;
    assert!(
        differing * 1000 <= pixels,
        "{input}: {differing} of {} channels differ from the OpenCV reference (worst delta {worst})",
        got.data.len()
    );
}

#[test]
fn rgb_image_matches_the_opencv_reference() {
    assert_matches_reference("photo.png", "photo_enhanced.png");
}

#[test]
fn rgba_image_matches_the_opencv_reference() {
    assert_matches_reference("photo_rgba.png", "photo_rgba_enhanced.png");
}

#[test]
fn greyscale_image_matches_the_opencv_reference() {
    assert_matches_reference("gray.png", "gray_enhanced.png");
}

#[test]
fn alpha_channel_is_passed_through_untouched() {
    let input = std::fs::read("tests/fixtures/photo_rgba.png").unwrap();
    let original = decode_rgba(&input);
    let enhanced = decode_rgba(&run(&input));

    for (pixel, (before, after)) in original
        .data
        .chunks_exact(4)
        .zip(enhanced.data.chunks_exact(4))
        .enumerate()
    {
        assert_eq!(before[3], after[3], "alpha changed at pixel {pixel}");
    }
}

#[test]
fn output_is_a_png() {
    let input = std::fs::read("tests/fixtures/photo.png").unwrap();
    let output = run(&input);
    assert_eq!(&output[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = decode_rgba(&output);
    assert_eq!((decoded.width, decoded.height), (40, 30));
}

#[test]
fn empty_stdin_is_an_error() {
    let output = Command::new(BIN)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to run adaptive-enhance");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no input on stdin"),
        "stderr was {stderr:?}"
    );
}

#[test]
fn invalid_input_is_an_error() {
    let mut child = Command::new(BIN)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"this is not a png")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("failed to decode PNG"));
}

#[test]
fn help_and_version() {
    let help = Command::new(BIN).arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("USAGE"));

    let version = Command::new(BIN).arg("--version").output().unwrap();
    assert!(version.status.success());
    assert!(String::from_utf8_lossy(&version.stdout).contains(env!("CARGO_PKG_VERSION")));
}
