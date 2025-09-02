use crate::matrix::Mat3d;

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

    pub fn into_matrix<const C: usize, const H: usize, const W: usize>(self) -> Mat3d<C, H, W> {
        assert_eq!(self.width * self.height * self.channels, self.pixels.len());
        Mat3d::from_vec(self.pixels)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}
