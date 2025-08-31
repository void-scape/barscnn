use std::io::Write;

use barscnn::image::prelude::*;

fn main() {
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let folder = make_filter_folder(time);
    let mut stats = make_statistics_file(time);

    const FILT1: usize = 8;
    let mut filt1: [_; FILT1] = [
        filter::vertical_sobel(),
        filter::horizontal_sobel(),
        filter::gaussian_blur_3x3(),
        filter::laplacian_3x3(),
        filter::identity(),
        filter::uniform_3x3(),
        filter::vertical_sobel(),
        filter::horizontal_sobel(),
    ];
    // const FILT2: usize = 4;
    // let mut filt2 = [filter::identity(); FILT2];

    const INPUT: usize = 1568;
    let mut fc1 = FcWeights::<INPUT, 26>::glorot();
    // let mut fc2 = FcWeights::<_, 64>::glorot();
    // let mut fc3 = FcWeights::<_, 26>::glorot();

    let mut total_iter = 0;
    for _ in 0.. {
        let mut cnn = ImageCnn::learning_rate(0.00001)
            .feature_map(&mut filt1)
            .relu()
            .max_pool(2)
            // .feature_map(&mut filt2)
            // .relu()
            // .max_pool(2)
            .flatten()
            .fully_connected(&mut fc1)
            .softmax();

        let iter = 1000;
        for _ in 0..iter {
            let (image, letter) = image_and_letter();
            let result = cnn.forward(&image);

            let index = letter as usize;
            cnn.backprop_index(index, result);
            total_iter += 1;
        }

        let mut acc = 0;
        let mut loss = 0.0;
        let test = 100;
        for _ in 0..test {
            let (image, letter) = image_and_letter();
            let index = letter as usize;

            let result = cnn.forward(&image);

            acc += (result
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0
                == index) as u32;
            loss += -result[index].ln();
        }

        // let (image, _) = image_and_letter();
        // let mut cnn = ImageCnn::learning_rate(0.00001)
        //     .feature_map(&mut filt1)
        //     .relu()
        //     .max_pool(2);
        // let result = cnn.forward(&image);
        // let mut image = result[0].clone();
        // for img in result.iter().skip(1) {
        //     image = image.stack(img);
        // }

        // let bytes = barscnn::image::bmp::to_bytes(image.as_bmp());
        // std::fs::write("target/out.bmp", bytes).unwrap();
        // std::process::Command::new("open")
        //     .arg("./target/out.bmp")
        //     .output()
        //     .unwrap();

        println!(
            "[Iterations {}]\tLoss: {:.2}\t | Accuracy: {:.2}%",
            total_iter,
            loss / test as f32,
            acc as f32 / test as f32 * 100.0,
        );

        save_filters(&folder, total_iter, &mut filt1);
        save_training_statistics(
            &mut stats,
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
    let bmp = barscnn::image::bmp::from_bytes(&bytes).unwrap();
    let image = barscnn::image::grayscale_from_bmp(&bmp).unwrap();
    assert_eq!(image.width(), 28);
    assert_eq!(image.height(), 28);

    (image, letter)
}

fn make_filter_folder(time: u64) -> String {
    let filter_folder = format!("./data/training/{}", time);
    std::fs::create_dir(&filter_folder).unwrap();
    filter_folder
}

fn save_filters(filter_folder: &str, iteration: usize, filters: &mut [Filter<9, 1, f32, f32>; 8]) {
    let (image, _) = image_and_letter();
    let mut test_cnn = ImageCnn::learning_rate(0.001).feature_map(filters);
    let result = test_cnn.forward(&image);
    let mut image = image.clone();
    for img in result.iter() {
        image = image.stack(img);
    }

    let bytes = barscnn::image::bmp::to_bytes(image.as_bmp());
    std::fs::write(format!("{}/{}.bmp", filter_folder, iteration), bytes).unwrap();

    let mut filter_string = String::new();
    for filter in filters.iter() {
        for y in 0..3 {
            for x in 0..3 {
                filter_string.push_str(&format!("{:.2} ", filter.weights()[0][y * 3 + x]));
            }
            filter_string.push_str("\n");
        }
        filter_string.push_str("\n");
    }

    let bytes = filter_string.as_bytes();
    std::fs::write(format!("{}/{}.txt", filter_folder, iteration), bytes).unwrap();
}

fn make_statistics_file(time: u64) -> std::fs::File {
    let path = format!("./data/training/{}/stats.csv", time);
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all("iteration,loss,accuracy\n".as_bytes())
        .unwrap();
    file
}

fn save_training_statistics(file: &mut std::fs::File, iteration: usize, loss: f32, accuracy: f32) {
    file.write_all(format!("{},{:.2},{:.2}\n", iteration, loss, accuracy).as_bytes())
        .unwrap();
}
