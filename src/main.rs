//! `adaptive-enhance` - read a PNG from stdin, write the enhanced PNG to stdout.
//!
//! ```text
//! adaptive-enhance < input.png > output.png
//! ```

use std::io::{Read, Write};

const HELP: &str = "\
adaptive-enhance - adaptive image enhancement (PNG in, PNG out)

USAGE:
    adaptive-enhance < input.png > output.png

Reads a PNG image from stdin, applies the adaptive enhancement
(HSV value channel combined from three vertical Gaussian scales, blended with
the weights of the principal component of (V1, V2)) and writes a PNG to stdout.

OPTIONS:
    -h, --help       Print this help and exit
    -V, --version    Print the version and exit

Supported inputs: 8/16-bit greyscale, greyscale+alpha, RGB, RGBA and palette
PNGs. The output is always 8-bit, RGB or RGBA (alpha is preserved).
";

fn main() {
    if let Err(message) = run() {
        eprintln!("adaptive-enhance: {message}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            "-V" | "--version" => {
                println!("adaptive-enhance {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "-" => {}
            other => {
                return Err(format!(
                    "unexpected argument '{other}' (the image is read from stdin and written to stdout)"
                ));
            }
        }
    }

    let mut input = Vec::new();
    std::io::stdin()
        .read_to_end(&mut input)
        .map_err(|e| format!("failed to read stdin: {e}"))?;
    if input.is_empty() {
        return Err("no input on stdin (usage: adaptive-enhance < input.png > output.png)".into());
    }

    let output = adaptive_enhance::png_io::enhance_png(&input)?;

    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(&output)
        .map_err(|e| format!("failed to write stdout: {e}"))?;
    stdout
        .flush()
        .map_err(|e| format!("failed to flush stdout: {e}"))?;
    Ok(())
}
