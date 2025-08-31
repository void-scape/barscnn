pub mod bmp;
pub mod pixel;

#[derive(Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub pixels: Vec<f32>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("channels", &self.channels)
            .finish_non_exhaustive()
    }
}

impl Image {
    pub fn shape(&self) -> Shape {
        Shape {
            width: self.width,
            height: self.height,
            channels: self.channels,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

// impl<T: Pixel> Image<T> {
//     pub fn width(&self) -> usize {
//         self.width as usize
//     }
//
//     pub fn height(&self) -> usize {
//         self.height as usize
//     }
//
//     pub fn as_bmp(&self) -> BmpWriter<'_, T> {
//         BmpWriter::new(self)
//     }
//
//     // pub fn stack<Other>(&self, other: &Image<Other>) -> Self
//     // where
//     //     Other: Pixel,
//     // {
//     //     assert_eq!(self.width, other.width);
//     //     let mut pixels = Vec::with_capacity(self.pixels.len() + other.pixels.len());
//     //     pixels.extend(self.pixels.iter());
//     //     pixels.extend(
//     //         other
//     //             .pixels
//     //             .iter()
//     //             .map(|p| T::from_linear_rgb(p.to_linear_rgb())),
//     //     );
//     //
//     //     Self {
//     //         width: self.width,
//     //         height: self.height + other.height,
//     //         pixels,
//     //     }
//     // }
// }
