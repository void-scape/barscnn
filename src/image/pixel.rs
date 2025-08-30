use super::layer::{CachedLayer, Layer};

#[derive(Debug)]
pub struct Pixels<T>(Vec<T>);

impl<T: Clone> Clone for Pixels<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Pixels<T> {
    pub fn new(pixels: Vec<T>) -> Self {
        Self(pixels)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
}

impl<T> std::ops::Index<usize> for Pixels<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> std::ops::IndexMut<usize> for Pixels<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PixelArray<const LEN: usize>([f32; LEN]);

impl<const LEN: usize> PixelArray<LEN> {
    pub fn new(pixels: [f32; LEN]) -> Self {
        Self(pixels)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.0.iter()
    }

    pub fn into_inner(self) -> [f32; LEN] {
        self.0
    }
}

impl<const LEN: usize> std::ops::Index<usize> for PixelArray<LEN> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const LEN: usize> std::ops::IndexMut<usize> for PixelArray<LEN> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const LEN: usize> Layer for PixelArray<LEN> {
    type Item = Self;
    type Cached = Self;

    fn forward(&self) -> Self::Item {
        PixelArray(self.into_inner())
    }

    fn forward_cached(self) -> CachedLayer<Self::Cached> {
        let item = self.forward();
        CachedLayer { layer: self, item }
    }
}

pub trait Pixel: Default + Copy {
    fn from_rgb(rgb: [u8; 3]) -> Self;
    fn to_rgb(self) -> [u8; 3];

    fn from_linear_rgb(rgb: [f32; 3]) -> Self;
    fn to_linear_rgb(self) -> [f32; 3];

    fn luminance(self) -> f32 {
        let rgb = self.to_linear_rgb();
        0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    }
}

pub type Grayscale = f32;

impl Pixel for f32 {
    fn from_rgb(rgb: [u8; 3]) -> Self {
        let r = rgb[0] as f32 / u8::MAX as f32;
        let g = rgb[1] as f32 / u8::MAX as f32;
        let b = rgb[2] as f32 / u8::MAX as f32;
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    fn to_rgb(self) -> [u8; 3] {
        let l = (self.clamp(0.0, 1.0) * u8::MAX as f32) as u8;
        [l, l, l]
    }

    fn from_linear_rgb(rgb: [f32; 3]) -> Self {
        0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    }

    fn to_linear_rgb(self) -> [f32; 3] {
        [self, self, self]
    }

    fn luminance(self) -> f32 {
        self
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Rgb(pub f32, pub f32, pub f32);

impl Pixel for Rgb {
    fn from_rgb(rgb: [u8; 3]) -> Self {
        let r = rgb[0] as f32 / u8::MAX as f32;
        let g = rgb[1] as f32 / u8::MAX as f32;
        let b = rgb[2] as f32 / u8::MAX as f32;
        Self(r, g, b)
    }

    fn to_rgb(self) -> [u8; 3] {
        let r = (self.0.clamp(0.0, 1.0) * u8::MAX as f32) as u8;
        let g = (self.1.clamp(0.0, 1.0) * u8::MAX as f32) as u8;
        let b = (self.2.clamp(0.0, 1.0) * u8::MAX as f32) as u8;
        [r, g, b]
    }

    fn from_linear_rgb(rgb: [f32; 3]) -> Self {
        Self(rgb[0], rgb[1], rgb[2])
    }

    fn to_linear_rgb(self) -> [f32; 3] {
        [self.0, self.1, self.2]
    }
}
