//! `adaptive-enhance` - read a PNG from stdin, write the enhanced PNG to stdout.
//!
//! ```text
//! adaptive-enhance < input.png > output.png
//! adaptive-enhance --adaptive --stats < input.png > output.png
//! ```

use std::io::{Read, Write};

const HELP: &str = "\
adaptive-enhance - contrast enhancement for low-light images (PNG in, PNG out)

USAGE:
    adaptive-enhance [OPTIONS] < input.png > output.png

Reads a PNG image from stdin and writes the enhanced PNG to stdout. The adaptive
enhancement is the illumination estimate at the core of both modes:

  * the exposure fusion framework, which synthesises a brighter exposure and
    decides per pixel how much of it to use (default);
  * the adaptive enhancement on its own (--adaptive).

OPTIONS:
        --fusion               Run the exposure fusion framework (default)
        --adaptive             Run the plain adaptive enhancement
    -k, --knee <VALUE>         Highlight-map knee: below it the map is the
                               identity, above it the synthesised exposure is
                               compressed toward 249/255. Higher is brighter and
                               flatter, lower is darker with more contrast.
                               When omitted, an auto-knee is computed from the
                               scene's diffuse highlight ceiling.
        --stats                Print statistics of the illumination map and the
                               chosen exposure ratio to stderr
    -h, --help                 Print this help and exit
    -V, --version              Print the version and exit

Supported inputs: 8/16-bit greyscale, greyscale+alpha, RGB, RGBA and palette
PNGs. The output is always 8-bit, RGB or RGBA (alpha is preserved).
";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Fusion,
    Adaptive,
}

struct Args {
    mode: Mode,
    knee: Option<f64>,
    stats: bool,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut args = Args {
        mode: Mode::Fusion,
        knee: None,
        stats: false,
    };

    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("adaptive-enhance {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "-" => {}
            "--fusion" => args.mode = Mode::Fusion,
            "--adaptive" => args.mode = Mode::Adaptive,
            "--stats" => args.stats = true,
            "-k" | "--knee" => {
                let value = argv.next().ok_or_else(|| format!("{arg} needs a value"))?;
                set_knee(&mut args, &value)?;
            }
            other => {
                if let Some(value) = other.strip_prefix("--knee=") {
                    set_knee(&mut args, value)?;
                } else {
                    return Err(format!(
                        "unexpected argument '{other}' (usage: adaptive-enhance [OPTIONS] < input.png > output.png)"
                    ));
                }
            }
        }
    }

    Ok(Some(args))
}

/// Set the highlight-map knee.
fn set_knee(args: &mut Args, value: &str) -> Result<(), String> {
    let knee: f64 = value
        .parse()
        .map_err(|_| format!("'{value}' is not a number (expected a knee between 0 and 1)"))?;
    if !(0.0..1.0).contains(&knee) {
        return Err(format!(
            "knee {value} is outside [0, 1): the map is applied to the synthesised \
             exposure, which is normalised, so its knee has to be inside the range"
        ));
    }
    args.knee = Some(knee);
    Ok(())
}

fn main() {
    if let Err(message) = run() {
        eprintln!("adaptive-enhance: {message}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = match parse_args()? {
        Some(args) => args,
        None => return Ok(()),
    };

    let mut input = Vec::new();
    std::io::stdin()
        .read_to_end(&mut input)
        .map_err(|e| format!("failed to read stdin: {e}"))?;
    if input.is_empty() {
        return Err("no input on stdin (usage: adaptive-enhance < input.png > output.png)".into());
    }

    let output = match args.mode {
        Mode::Fusion => {
            let image = adaptive_enhance_fusion::png_io::decode_png(&input)?;
            let params = adaptive_enhance_fusion::pipeline::PipelineParams {
                fusion: adaptive_enhance_fusion::fusion::FusionParams {
                    knee: args.knee,
                    ..Default::default()
                },
            };
            let fused = adaptive_enhance_fusion::pipeline::enhance_rgb(
                &params,
                &image.rgb,
                image.width,
                image.height,
            );
            if args.stats {
                let stats = adaptive_enhance_fusion::pipeline::map_stats(&fused.illumination);
                eprintln!(
                    "illumination: adaptive-enhancement\n  min {:.4} max {:.4} mean {:.4}\n  under-exposed {:.2}%\n  exposure ratio {:.4}",
                    stats.min,
                    stats.max,
                    stats.mean,
                    stats.under_exposed_fraction * 100.0,
                    fused.exposure_ratio
                );
                if let Some(explicit) = args.knee {
                    eprintln!("knee: {:.2}", explicit);
                } else {
                    eprintln!("knee: auto ({:.2})", fused.effective_knee);
                }
            }
            adaptive_enhance_fusion::png_io::encode_png(&adaptive_enhance_fusion::png_io::Image {
                width: image.width,
                height: image.height,
                rgb: fused.rgb,
                alpha: image.alpha,
            })?
        }
        Mode::Adaptive => adaptive_enhance_fusion::png_io::enhance_png(&input)?,
    };

    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(&output)
        .map_err(|e| format!("failed to write stdout: {e}"))?;
    stdout
        .flush()
        .map_err(|e| format!("failed to flush stdout: {e}"))?;
    Ok(())
}
