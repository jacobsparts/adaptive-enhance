use adaptive_enhance_fusion::pipeline::{enhance_rgb, PipelineParams};

#[test]
fn test_smoke_fusion() {
    let width = 64;
    let height = 64;
    let rgb = vec![128u8; width * height * 3];
    let params = PipelineParams::default();
    let out = enhance_rgb(&params, &rgb, width, height);
    assert_eq!(out.rgb.len(), width * height * 3);
}

#[test]
fn test_smoke_adaptive() {
    let width = 64;
    let height = 64;
    let rgb = vec![128u8; width * height * 3];
    let out = adaptive_enhance_fusion::adaptive_enhance_rgb(&rgb, width, height);
    assert_eq!(out.len(), width * height * 3);
}

#[test]
fn test_smoke_white_balance_png() {
    use adaptive_enhance_fusion::png_io;
    use adaptive_enhance_fusion::white_balance::{self, Params};

    let width = 64;
    let height = 64;
    // A tinted border around a dark centre, so the white point comes from the
    // border and the correction has something to do.
    let mut rgb = vec![0u8; width * height * 3];
    for pixel in 0..width * height {
        rgb[3 * pixel] = 180;
        rgb[3 * pixel + 1] = 175;
        rgb[3 * pixel + 2] = 160;
    }
    let input = png_io::encode_png(&png_io::Image {
        width,
        height,
        rgb,
        alpha: None,
    })
    .expect("encode");

    for params in [Params::default(), Params::safe(0.5)] {
        let output = white_balance::white_balance_png(&input, &params).expect("white balance");
        let image = png_io::decode_png(&output).expect("decode");
        assert_eq!(image.width, width);
        assert_eq!(image.height, height);
        assert_eq!(image.rgb.len(), width * height * 3);
        assert!(image.alpha.is_none());
    }
}
