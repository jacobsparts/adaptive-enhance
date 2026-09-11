//! PNG decoding/encoding helpers.
//!
//! The command line tool reads a PNG on stdin and writes a PNG on stdout; the
//! helpers live here so that they can be exercised by the integration tests.

use std::io::{Cursor, Write};

use png::{BitDepth, ColorType, Transformations};

/// An 8-bit image, interleaved RGB, with an optional separate alpha plane.
pub struct Image {
    pub width: usize,
    pub height: usize,
    /// `width * height * 3` bytes, R, G, B.
    pub rgb: Vec<u8>,
    /// `width * height` bytes, present only if the input had an alpha channel.
    pub alpha: Option<Vec<u8>>,
}

impl Image {
    /// Number of pixels.
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    /// True if the image has no pixels.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Interleaved RGB or RGBA samples, ready to be encoded as a PNG.
    pub fn to_interleaved(&self) -> Vec<u8> {
        match &self.alpha {
            None => self.rgb.clone(),
            Some(alpha) => {
                let mut out = Vec::with_capacity(self.len() * 4);
                for (rgb, a) in self.rgb.chunks_exact(3).zip(alpha) {
                    out.extend_from_slice(rgb);
                    out.push(*a);
                }
                out
            }
        }
    }
}

/// Decode a PNG image into 8-bit RGB (+ alpha if present).
///
/// Palette images and sub-byte greyscale are expanded by the decoder; 16-bit
/// samples are reduced to their most significant byte.
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

    // Extract one 8-bit channel of every pixel (high byte for 16-bit samples).
    let channel = |index: usize| -> Vec<u8> {
        (0..pixels)
            .map(|p| buffer[(p * channels + index) * sample_bytes])
            .collect()
    };

    let rgb = match info.color_type {
        ColorType::Rgb | ColorType::Rgba => (0..pixels)
            .flat_map(|pixel| {
                [
                    buffer[(pixel * channels) * sample_bytes],
                    buffer[(pixel * channels + 1) * sample_bytes],
                    buffer[(pixel * channels + 2) * sample_bytes],
                ]
            })
            .collect(),
        ColorType::Grayscale | ColorType::GrayscaleAlpha => {
            channel(0).into_iter().flat_map(|g| [g, g, g]).collect()
        }
        ColorType::Indexed => unreachable!("handled above"),
    };

    let alpha = match info.color_type {
        ColorType::Rgba | ColorType::GrayscaleAlpha => Some(channel(channels - 1)),
        _ => None,
    };

    Ok(Image {
        width,
        height,
        rgb,
        alpha,
    })
}

/// Encode an 8-bit RGB/RGBA image as a PNG.
pub fn encode_png(image: &Image) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, image.width as u32, image.height as u32);
        encoder.set_color(if image.alpha.is_some() {
            ColorType::Rgba
        } else {
            ColorType::Rgb
        });
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

/// Decode a PNG, apply the adaptive enhancement and re-encode it as a PNG.
pub fn enhance_png(input: &[u8]) -> Result<Vec<u8>, String> {
    let image = decode_png(input)?;
    let enhanced = crate::adaptive_enhance_rgb(&image.rgb, image.width, image.height);
    encode_png(&Image {
        width: image.width,
        height: image.height,
        rgb: enhanced,
        alpha: image.alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_16_bit(color: ColorType, width: u32, data: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut output, width, 1);
            encoder.set_color(color);
            encoder.set_depth(BitDepth::Sixteen);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(data).unwrap();
        }
        output
    }

    #[test]
    fn decodes_16_bit_rgb_using_most_significant_bytes() {
        let input = encode_16_bit(
            ColorType::Rgb,
            2,
            &[
                0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44,
            ],
        );

        let image = decode_png(&input).unwrap();

        assert_eq!(image.rgb, [0x12, 0x56, 0x9a, 0xde, 0x11, 0x33]);
        assert_eq!(image.alpha, None);
    }

    #[test]
    fn decodes_16_bit_rgba_and_preserves_alpha() {
        let input = encode_16_bit(
            ColorType::Rgba,
            1,
            &[0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0],
        );

        let image = decode_png(&input).unwrap();

        assert_eq!(image.rgb, [0x12, 0x56, 0x9a]);
        assert_eq!(image.alpha, Some(vec![0xde]));
    }
}
