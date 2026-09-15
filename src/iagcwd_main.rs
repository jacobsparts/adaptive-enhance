//! `iagcwd` - improved adaptive gamma correction, PNG in and PNG out.
//!
//! ```text
//! iagcwd < input.png > output.png
//! iagcwd --tau-t 0.4 --stats < input.png > output.png
//! ```
//!
//! This is the companion to `adaptive-enhance`: the exposure fusion lifts
//! underexposed scenes, this pulls down over-bright ones.

use std::io::{Read, Write};

use adaptive_enhance_fusion::gray_png;
use adaptive_enhance_fusion::iagcwd::{self, Mode, Params};

const HELP: &str = "\
iagcwd - improved adaptive gamma correction for brightness-distorted images
         (PNG in, PNG out)

USAGE:
    iagcwd [OPTIONS] < input.png > output.png

Reads a PNG image from stdin and writes the corrected PNG to stdout. The
intensity of the image is measured against an expected average and corrected
with the weighted histogram of that intensity: a dimmed image is lifted, a
bright one is pulled down, and one already close to the expected average is
copied through unchanged. On a colour image only the value channel is
corrected, so hue and saturation survive; a greyscale image stays greyscale.

OPTIONS:
        --dim-alpha <VALUE>     Weighting exponent of the dimmed path
                                (default: 0.75; lower is stronger)
        --bright-alpha <VALUE>  Weighting exponent of the bright path
                                (default: 0.25; lower is stronger)
        --target <VALUE>        Expected average intensity, 0..255
                                (default: 112)
        --tau-t <VALUE>         Relative deviation from the target that marks an
                                image as dimmed or bright (default: 0.3)
        --tau <VALUE>           Floor of the inverse CDF on the bright path,
                                0..1 (default: 0.5)
        --mode <MODE>           Force a path instead of deciding from the
                                image: dimmed, bright or none
        --stats                 Print the decision and the correction statistics
                                to stderr
    -h, --help                  Print this help and exit
    -V, --version               Print the version and exit

Supported inputs: 8/16-bit greyscale, greyscale+alpha, RGB, RGBA and palette
PNGs. The output is always 8-bit, and keeps the input's colour type (alpha is
preserved).
";

/// A forced path, as `--mode` spells it.
fn parse_mode(value: &str) -> Result<Mode, String> {
    match value {
        "dimmed" | "dim" => Ok(Mode::Dimmed),
        "bright" => Ok(Mode::Bright),
        "none" | "off" | "unchanged" => Ok(Mode::Unchanged),
        other => Err(format!(
            "'{other}' is not a mode (expected dimmed, bright or none)"
        )),
    }
}

struct Args {
    params: Params,
    mode: Option<Mode>,
    stats: bool,
}

fn parse_number(flag: &str, value: &str) -> Result<f64, String> {
    value
        .parse()
        .map_err(|_| format!("{flag}: '{value}' is not a number"))
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut args = Args {
        params: Params::default(),
        mode: None,
        stats: false,
    };

    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        let mut value = |flag: &str| -> Result<String, String> {
            argv.next().ok_or_else(|| format!("{flag} needs a value"))
        };
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("iagcwd {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "-" => {}
            "--stats" => args.stats = true,
            "--dim-alpha" => {
                args.params.alpha_dimmed = parse_number("--dim-alpha", &value("--dim-alpha")?)?
            }
            "--bright-alpha" => {
                args.params.alpha_bright =
                    parse_number("--bright-alpha", &value("--bright-alpha")?)?
            }
            "--target" => args.params.target = parse_number("--target", &value("--target")?)?,
            "--tau-t" => args.params.tau_t = parse_number("--tau-t", &value("--tau-t")?)?,
            "--tau" => args.params.tau = parse_number("--tau", &value("--tau")?)?,
            "--mode" => args.mode = Some(parse_mode(&value("--mode")?)?),
            other => {
                let (flag, inline) = match other.split_once('=') {
                    Some((flag, value)) => (flag, Some(value.to_string())),
                    None => (other, None),
                };
                let mut take = |flag: &str| -> Result<String, String> {
                    match &inline {
                        Some(value) => Ok(value.clone()),
                        None => value(flag),
                    }
                };
                match flag {
                    "--dim-alpha" => args.params.alpha_dimmed = parse_number(flag, &take(flag)?)?,
                    "--bright-alpha" => args.params.alpha_bright = parse_number(flag, &take(flag)?)?,
                    "--target" => args.params.target = parse_number(flag, &take(flag)?)?,
                    "--tau-t" => args.params.tau_t = parse_number(flag, &take(flag)?)?,
                    "--tau" => args.params.tau = parse_number(flag, &take(flag)?)?,
                    "--mode" => args.mode = Some(parse_mode(&take(flag)?)?),
                    _ => {
                        return Err(format!(
                            "unexpected argument '{other}' (usage: iagcwd [OPTIONS] < input.png > output.png)"
                        ))
                    }
                }
            }
        }
    }

    if args.params.target <= 0.0 {
        return Err("--target must be positive: it is the expected average intensity".into());
    }
    if args.params.tau_t < 0.0 {
        return Err("--tau-t must not be negative".into());
    }
    if !(0.0..=1.0).contains(&args.params.tau) {
        return Err("--tau must be inside [0, 1]".into());
    }
    Ok(Some(args))
}

fn main() {
    if let Err(message) = run() {
        eprintln!("iagcwd: {message}");
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
        return Err("no input on stdin (usage: iagcwd < input.png > output.png)".into());
    }

    let image = gray_png::decode_png(&input)?;
    let params = args.params;
    let (enhanced, info) = gray_png::enhance_image(image, &params, args.mode);
    if args.stats {
        eprintln!(
            "iagcwd: {}\n  mean {:.4} of {:.1} -> {:.4}\n  alpha {:.4}, truncated {}, inverted {}\n  levels {} of 256, table[{}] {}",
            info.mode.label(),
            info.mean_intensity,
            params.target,
            info.deviation,
            info.alpha,
            info.truncated,
            info.inverted,
            info.levels,
            iagcwd::TAB_LOW,
            info.table_low
        );
    }
    let output = gray_png::encode_png(&enhanced)?;

    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(&output)
        .map_err(|e| format!("failed to write stdout: {e}"))?;
    stdout
        .flush()
        .map_err(|e| format!("failed to flush stdout: {e}"))?;
    Ok(())
}
