use barscnn::image;

fn main() {
    let bytes = std::fs::read("data/letters/A/A-0.bmp").unwrap();
    let bmp = image::bmp::from_bytes(&bytes).unwrap();

    let image = image::grayscale_from_bmp(&bmp).unwrap();
    assert_eq!(image.width(), 28);
    assert_eq!(image.height(), 28);

    for _ in 0..5 {
        let filters = [image::filter::gaussian_blur_3x3(); 8];
        const INPUT: usize = 14 * 14 * 8;
        const OUTPUT: usize = 26;
        let fc = image::linear::FullyConnected::<INPUT, OUTPUT>::glorot();

        let iter = 100;
        let mut acc = 0;
        let mut loss = 0.0;
        for _ in 0..iter {
            let result = image
                .filter_features(&filters)
                .max_pool(2)
                .flatten::<INPUT>()
                .fully_connected(&fc)
                .softmax();
            acc += (result
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0
                == 0) as u32;
            loss += -result[0].ln();
        }
        println!(
            "[Iterations {}]\tAccuracy: {:.2}%\t| Loss: {:.2}",
            iter,
            acc as f32 / iter as f32 * 100.0,
            loss / iter as f32,
        );
    }
}
