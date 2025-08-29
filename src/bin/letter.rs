use barscnn::image;

fn main() {
    let bytes = std::fs::read("data/letters/A/A-0.bmp").unwrap();
    let bmp = image::bmp::from_bytes(&bytes).unwrap();

    let image = image::grayscale_from_bmp(&bmp).unwrap();
    assert_eq!(image.width(), 28);
    assert_eq!(image.height(), 28);
    let filter = image::filter::gaussian_blur_3x3();

    let filtered = filter.conv_padded(&image);
    let pooled = image::pool::max_pool(2, &filtered);

    const INPUT: usize = 14 * 14;
    const OUTPUT: usize = 26;

    let pixels = pooled.flatten::<INPUT>();
    let reduce = image::linear::Reduce::new([[0.0; INPUT]; OUTPUT], [0.0; OUTPUT]);

    let reduced = reduce.forward(&pixels);
    let softmax = image::linear::softmax(&reduced);

    println!("{softmax:#?}");
}
