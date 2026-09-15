//! PNG decoding and encoding for the gamma-correction binary.
//!
//! Unlike [`crate::png_io`], which always works on colour, this keeps a
//! greyscale image greyscale: a corrected grey image never picks up colour
//! channels it did not have.

use std::io::{Cursor, Write};

use png::{BitDepth, ColorType, Transformations};

/// An 8-bit image: interleaved RGB, or a single intensity plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    /// Interleaved RGB samples on a colour image.
    pub rgb: Option<Vec<u8>>,
    /// Intensity samples of a greyscale image, `width * height` bytes.
    pub gray: Option<Vec<u8>>,
    /// `width * height` bytes, present only if the input had an alpha channel.
    pub alpha: Option<Vec<u8>>,
}

impl Image {
    /// Number of pixels.
    pub fn pixels(&self) -> usize {
        self.width * self.height
    }

    /// Whether the image carries an intensity plane instead of colour.
    pub fn is_gray(&self) -> bool {
        self.gray.is_some()
    }

    /// Interleaved samples in the channel layout the encoder writes.
    pub fn to_interleaved(&self) -> Vec<u8> {
        match (&self.gray, &self.rgb) {
            (Some(gray), _) => match &self.alpha {
                None => gray.clone(),
                Some(alpha) => gray.iter().zip(alpha).flat_map(|(g, a)| [*g, *a]).collect(),
            },
            (None, Some(rgb)) => match &self.alpha {
                None => rgb.clone(),
                Some(alpha) => rgb
                    .chunks_exact(3)
                    .zip(alpha)
                    .flat_map(|(rgb, a)| [rgb[0], rgb[1], rgb[2], *a])
                    .collect(),
            },
            (None, None) => Vec::new(),
        }
    }
}

/// Decode a PNG image into 8-bit RGB (or grey) plus alpha when present.
///
/// Palette images and sub-byte greyscale are expanded by the decoder; 16-bit
/// samples are reduced to their most significant byte. RGBA input is decoded as
/// RGB with the alpha kept aside, whether or not the alpha is opaque, so a
/// fully transparent colour still has its colour corrected.
pub fn decode_png(input: &[u8]) -> Result<Image, String> {
    let mut decoder = png::Decoder::new(Cursor::new(input));
    decoder.set_transformations(Transformations::EXPAND);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("failed to decode PNG: {e}"))?;

    let mut buffer = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|e| format!("failed to decode PNG: {e}"))?;
    buffer.truncate(info.buffer_size());

    let width = info.width as usize;
    let height = info.height as usize;
    if width == 0 || height == 0 {
        return Err("PNG has zero width or height".into());
    }

    let sample_bytes = match info.bit_depth {
        BitDepth::Eight => 1,
        BitDepth::Sixteen => 2,
        other => return Err(format!("unsupported PNG bit depth {other:?}")),
    };
    let channels = match info.color_type {
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Indexed => {
            return Err("unsupported PNG colour type Indexed (palette expansion failed)".into())
        }
    };
    let pixels = width * height;

    // One 8-bit sample per channel of every pixel (high byte for 16-bit).
    let channel = |index: usize| -> Vec<u8> {
        (0..pixels)
            .map(|p| buffer[(p * channels + index) * sample_bytes])
            .collect()
    };

    let rgb = match info.color_type {
        ColorType::Rgb | ColorType::Rgba => Some(
            (0..pixels)
                .flat_map(|pixel| {
                    [
                        buffer[(pixel * channels) * sample_bytes],
                        buffer[(pixel * channels + 1) * sample_bytes],
                        buffer[(pixel * channels + 2) * sample_bytes],
                    ]
                })
                .collect(),
        ),
        ColorType::Grayscale | ColorType::GrayscaleAlpha => None,
        ColorType::Indexed => unreachable!("handled above"),
    };
    let gray = match info.color_type {
        ColorType::Grayscale | ColorType::GrayscaleAlpha => Some(channel(0)),
        _ => None,
    };
    let alpha = match info.color_type {
        ColorType::Rgba | ColorType::GrayscaleAlpha => Some(channel(channels - 1)),
        _ => None,
    };

    Ok(Image {
        width,
        height,
        rgb,
        gray,
        alpha,
    })
}

/// Encode an 8-bit image as a PNG.
pub fn encode_png(image: &Image) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let color = match (image.gray.is_some(), image.alpha.is_some()) {
        (true, false) => ColorType::Grayscale,
        (true, true) => ColorType::GrayscaleAlpha,
        (false, false) => ColorType::Rgb,
        (false, true) => ColorType::Rgba,
    };
    {
        let mut encoder = png::Encoder::new(&mut out, image.width as u32, image.height as u32);
        encoder.set_color(color);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("failed to write PNG header: {e}"))?;
        writer
            .write_image_data(&image.to_interleaved())
            .map_err(|e| format!("failed to write PNG data: {e}"))?;
    }
    out.flush()
        .map_err(|e| format!("failed to finish PNG: {e}"))?;
    Ok(out)
}

/// Run the enhancement on a decoded image, keeping its channel layout.
///
/// `mode` overrides the decision [`crate::iagcwd::Params::mode`] would make,
/// which is what the command's `--mode` flag exposes.
pub fn enhance_image(
    image: Image,
    params: &crate::iagcwd::Params,
    mode: Option<crate::iagcwd::Mode>,
) -> (Image, crate::iagcwd::Info) {
    let mut image = image;
    let info = match (&image.rgb, &image.gray) {
        (Some(rgb), None) => {
            let (rgb, info) =
                crate::iagcwd::enhance_rgb(rgb, image.width, image.height, params, mode);
            image.rgb = Some(rgb);
            info
        }
        (None, Some(gray)) => {
            let mut gray = gray.clone();
            let info =
                crate::iagcwd::enhance_gray(&mut gray, image.width, image.height, params, mode);
            image.gray = Some(gray);
            info
        }
        _ => unreachable!("an image carries exactly one of rgb and gray"),
    };
    (image, info)
}
