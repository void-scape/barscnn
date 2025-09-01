pub mod bmp;

#[derive(Debug, Default, Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub pixels: Vec<f32>,
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
