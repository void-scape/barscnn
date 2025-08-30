use barscnn::image::feature::FeatureMapData;
use barscnn::image::flatten::FlattenData;
use barscnn::image::layer::Layer;
use barscnn::image::linear::{FullyConnectedData, SoftmaxData};
use barscnn::image::pixel::Grayscale;
use barscnn::image::pool::MaxPoolData;
use barscnn::image::{self, Image};

fn main() {
    const FILTERS: usize = 16;
    let mut filters = [image::filter::uniform_3x3(); FILTERS];
    const INPUT: usize = 14 * 14 * FILTERS;
    const OUTPUT: usize = 26;
    let mut fc = image::linear::FcWeights::<INPUT, OUTPUT>::glorot();

    let mut total_iter = 0;
    for _ in 0.. {
        let iter = 300;
        for _ in 0..iter {
            let (image, letter) = image_and_letter();
            let mut cnn = image
                .feature_map(&mut filters)
                .max_pool(2)
                .flatten()
                .fully_connected(&mut fc)
                .softmax();
            let result = cnn.forward();
            let index = letter as usize;
            cnn.backprop_index(index, result, 0.001);
            total_iter += 1;
        }

        let mut acc = 0;
        let mut loss = 0.0;
        let test = 200;
        for _ in 0..test {
            let (image, letter) = image_and_letter();
            let index = letter as usize;

            let mut cnn = image
                .feature_map(&mut filters)
                .max_pool(2)
                .flatten()
                .fully_connected(&mut fc)
                .softmax();
            let result = cnn.forward();

            acc += (result
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0
                == index) as u32;
            loss += -result[index].ln();
        }

        println!(
            "[Iterations {}]\tLoss: {:.2}\t | Accuracy: {:.2}%",
            total_iter,
            loss / test as f32,
            acc as f32 / test as f32 * 100.0,
        );
    }
}

fn image_and_letter() -> (Image<Grayscale>, u8) {
    let letter = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u8
        % 26;
    let ascii_letter = ('A' as u8 + letter) as char;

    let index = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as usize
        % 1032;

    let bytes = std::fs::read(format!(
        "data/letters/{}/{}-{}.bmp",
        ascii_letter, ascii_letter, index
    ))
    .unwrap();
    let bmp = image::bmp::from_bytes(&bytes).unwrap();
    let image = image::grayscale_from_bmp(&bmp).unwrap();
    assert_eq!(image.width(), 28);
    assert_eq!(image.height(), 28);

    (image, letter)
}
