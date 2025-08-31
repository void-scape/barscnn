pub mod bmp;

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
