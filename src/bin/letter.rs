use barscnn::image;
use barscnn::image::feature::FeatureMapImage;
use barscnn::image::flatten::FlattenData;
use barscnn::image::layer::Layer;
use barscnn::image::linear::{FullyConnectedData, SoftmaxData};
use barscnn::image::pool::MaxPoolData;

fn main() {
    let bytes = std::fs::read("data/letters/A/A-0.bmp").unwrap();
    let bmp = image::bmp::from_bytes(&bytes).unwrap();

    let image = image::grayscale_from_bmp(&bmp).unwrap();
    assert_eq!(image.width(), 28);
    assert_eq!(image.height(), 28);

    for _ in 0..5 {
        let iter = 100;
        let mut acc = 0;
        let mut loss = 0.0;
        for _ in 0..iter {
            let filters = [image::filter::gaussian_blur_3x3(); 8];
            const INPUT: usize = 14 * 14 * 8;
            const OUTPUT: usize = 26;
            let fc = image::linear::FcWeights::<INPUT, OUTPUT>::glorot();

            let mut cnn = image
                .feature_map(&filters)
                .max_pool(2)
                .flatten()
                .fully_connected(&fc)
                .softmax();
            let result = cnn.forward();

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
