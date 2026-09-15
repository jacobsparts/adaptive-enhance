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
