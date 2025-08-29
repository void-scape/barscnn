use barscnn::image;

fn main() {
    let bytes = std::fs::read("assets/mandelbrot.bmp").unwrap();
    let bmp = image::bmp::from_bytes(&bytes).unwrap();

    let image = image::rgb_from_bmp(&bmp).unwrap();
    let filter = image::filter::vertical_sobel();

    let output = filter.conv_padded(&image);
    let bytes = image::bmp::to_bytes(image.stack(&output).as_bmp());
    std::fs::write("target/out.bmp", bytes).unwrap();

    std::process::Command::new("open")
        .arg("./target/out.bmp")
        .output()
        .unwrap();
}
