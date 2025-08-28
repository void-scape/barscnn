pub trait Pixel: Default + Copy {
    fn from_rgb(rgb: [u8; 3]) -> Self;
    fn to_rgb(self) -> [u8; 3];

    fn from_linear_rgb(rgb: [f32; 3]) -> Self;
    fn to_linear_rgb(self) -> [f32; 3];
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Grayscale(pub f32);

impl Pixel for Grayscale {
    fn from_rgb(rgb: [u8; 3]) -> Self {
        let r = rgb[0] as f32 / u8::MAX as f32;
        let g = rgb[1] as f32 / u8::MAX as f32;
        let b = rgb[2] as f32 / u8::MAX as f32;
        Self(0.2126 * r + 0.7152 * g + 0.0722 * b)
    }

    fn to_rgb(self) -> [u8; 3] {
        let l = (self.0.clamp(0.0, 1.0) * u8::MAX as f32) as u8;
        [l, l, l]
    }

    fn from_linear_rgb(rgb: [f32; 3]) -> Self {
        Self(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    }

    fn to_linear_rgb(self) -> [f32; 3] {
        [self.0, self.0, self.0]
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
