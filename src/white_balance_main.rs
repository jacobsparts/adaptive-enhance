//! `white-balance` - automatic white balance for product photos (PNG in, PNG out).
//!
//! ```text
//! white-balance < input.png > output.png
//! white-balance --safe --knee 0.7 < input.png > output.png
//! ```

use std::io::{Read, Write};

use adaptive_enhance_fusion::png_io;
use adaptive_enhance_fusion::white_balance::{self, Params};

const HELP: &str = "\
white-balance - automatic white balance for photographs on a white background
                (PNG in, PNG out)

USAGE:
    white-balance [OPTIONS] < input.png > output.png

Reads a PNG image from stdin and writes the white-balanced PNG to stdout. The
white point is measured from the outer 5% of the shorter side, which is where a
product photo's background is, and each channel is scaled so that white point
lands on 255. The subject cannot drag the estimate and neither can a shadow on
the sweep.

By default every level is scaled by the same per-channel gain and the top of the
range clips. With --safe the top of the range is rolled gently into 255 instead,
so a specular highlight keeps its shape.

OPTIONS:
    -s, --safe                 Soften the highlights instead of clipping them
    -k, --knee <VALUE>         Point on the 0..=1 range where --safe starts to
                               roll off (default: 0.5). In the plain mode the
                               value is the clip level applied to the input
                               (default: 1.0, no clip)
        --stats                Print the measured white point and the fraction
                               of clipped samples to stderr
    -h, --help                 Print this help and exit
    -V, --version              Print the version and exit

Supported inputs: 8/16-bit greyscale, greyscale+alpha, RGB, RGBA and palette
PNGs. The output is always 8-bit and keeps the input's colour type (alpha is
preserved).
";

struct Args {
    params: Params,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut params = Params::default();
    let mut safe = false;
    let mut knee: Option<f64> = None;

    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("white-balance {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "-" => {}
            "-s" | "--safe" => safe = true,
            "-k" | "--knee" => {
                let value = argv.next().ok_or_else(|| format!("{arg} needs a value"))?;
                knee = Some(parse_knee(&value)?);
            }
            other => {
                if let Some(value) = other.strip_prefix("--knee=") {
                    knee = Some(parse_knee(value)?);
                } else {
                    return Err(format!(
                        "unexpected argument '{other}' (usage: white-balance [OPTIONS] < input.png > output.png)"
                    ));
                }
            }
        }
    }

    if safe {
        params = Params::safe(knee.unwrap_or(white_balance::DEFAULT_KNEE));
    } else if let Some(knee) = knee {
        params.knee = knee;
    }
    Ok(Some(Args { params }))
}

fn parse_knee(value: &str) -> Result<f64, String> {
    let knee: f64 = value
        .parse()
        .map_err(|_| format!("'{value}' is not a number (expected a knee between 0 and 1)"))?;
    if !(0.0..=1.0).contains(&knee) {
        return Err(format!(
            "knee {value} is outside [0, 1]: it is a point on the normalised range"
        ));
    }
    Ok(knee)
}

fn main() {
    if let Err(message) = run() {
        eprintln!("white-balance: {message}");
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
        return Err("no input on stdin (usage: white-balance < input.png > output.png)".into());
    }

    let params = args.params;
    let mut image = png_io::decode_png(&input)?;
    let info = white_balance::correct_image(&mut image, &params);
    if args.params.safe {
        eprintln!("white-balance: safe, knee {:.3}", params.knee);
    }
    eprintln!(
        "white-balance: white point {:.1} {:.1} {:.1}, gains {:.4} {:.4} {:.4}, {:.2}% clipped",
        info.white.channels[0],
        info.white.channels[1],
        info.white.channels[2],
        info.white.gains[0],
        info.white.gains[1],
        info.white.gains[2],
        100.0 * info.clipped_fraction,
    );
    let output = png_io::encode_png(&image)?;

    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(&output)
        .map_err(|e| format!("failed to write stdout: {e}"))?;
    stdout
        .flush()
        .map_err(|e| format!("failed to flush stdout: {e}"))?;
    Ok(())
}
