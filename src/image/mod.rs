use self::bmp::{BmpReader, BmpWriter};
use self::pixel::{Grayscale, Pixel, PixelArray, Pixels, Rgb};

pub mod bmp;
pub mod filter;
pub mod linear;
pub mod pixel;
pub mod pool;

pub fn rgb_from_bmp(bmp: &BmpReader) -> Result<Image<Rgb>, &'static str> {
    rgb_image_from_bmp(bmp)
}

pub fn grayscale_from_bmp(bmp: &BmpReader) -> Result<Image<Grayscale>, &'static str> {
    grayscale_image_from_bmp(bmp)
}

#[derive(Debug)]
pub struct Image<T> {
    width: u32,
    height: u32,
    pixels: Pixels<T>,
}

impl<T: Pixel> Image<T> {
    pub fn width(&self) -> usize {
        self.width as usize
    }

    pub fn height(&self) -> usize {
        self.height as usize
    }

    pub fn as_bmp(&self) -> BmpWriter<'_, T> {
        BmpWriter::new(self)
    }

    pub fn stack<Other>(&self, other: &Image<Other>) -> Self
    where
        Other: Pixel,
    {
        assert_eq!(self.width, other.width);
        let mut pixels = Vec::with_capacity(self.pixels.len() + other.pixels.len());
        pixels.extend(self.pixels.iter());
        pixels.extend(
            other
                .pixels
                .iter()
                .map(|p| T::from_linear_rgb(p.to_linear_rgb())),
        );

        Self {
            width: self.width,
            height: self.height + other.height,
            pixels: Pixels::new(pixels),
        }
    }

    pub fn pixels(&self) -> impl Iterator<Item = T> {
        self.pixels.iter().copied()
    }
}

impl Image<Grayscale> {
    pub fn flatten<const LEN: usize>(&self) -> PixelArray<LEN> {
        if self.pixels.len() != LEN {
            panic!(
                "called `Image::flatten` with the incorrect `LEN`, expected {}, got {LEN}",
                self.pixels.len()
            );
        }

        let mut arr = [0.0; LEN];
        arr.copy_from_slice(self.pixels.as_slice());
        PixelArray::new(arr)
    }
}

fn rgb_image_from_bmp(bmp: &BmpReader) -> Result<Image<Rgb>, &'static str> {
    let mut pixels = Vec::with_capacity((bmp.width * bmp.height) as usize);

    fn read_slice<const LEN: usize>(slice: &mut &[u8]) -> [u8; LEN] {
        let mut data = [0; LEN];
        data.copy_from_slice(&slice[0..LEN]);
        *slice = &slice[LEN..];
        data
    }

    match bmp.compression {
        bmp::Compression::Rgb => {
            debug_assert_eq!(bmp.bpp, 24);

            let bpr = bmp.width as usize * 3;
            let padding = (4 - (bpr % 4)) % 4;

            let bytes = bmp.pixels;
            let input = &mut &bytes[..];
            for h in (0..bmp.height as usize).rev() {
                *input = &bytes[h * 3 * bmp.width as usize + padding..];

                for _ in 0..bmp.width {
                    let b = input[0];
                    let g = input[1];
                    let r = input[2];
                    *input = &input[3..];
                    pixels.push(Rgb::from_rgb([r, g, b]));
                }
            }
        }
        bmp::Compression::BitFields { r, g, b } => {
            let row_size = ((bmp.bpp * bmp.width) as f32 / 32.0).ceil() as usize;
            debug_assert_eq!(bmp.width as usize, row_size);
            debug_assert_eq!(bmp.bpp, 32);

            fn u32(input: &mut &[u8]) -> Result<u32, &'static str> {
                if input.len() < 4 {
                    return Err("Unexpected end of file");
                }
                Ok(u32::from_le_bytes(read_slice::<4>(input)))
            }

            fn extract_channel(pixel: u32, mask: u32) -> u8 {
                if mask == 0 {
                    return 0;
                }
                let shift = mask.trailing_zeros();
                let value = (pixel & mask) >> shift;
                let max_value = mask >> shift;
                ((value * 255) / max_value) as u8
            }

            let bytes = bmp.pixels;
            let input = &mut &bytes[..];
            for h in (0..bmp.height as usize).rev() {
                *input = &bytes[h * 4 * bmp.width as usize..];

                for _ in 0..bmp.width {
                    let pixel = u32(input)?;
                    let r = extract_channel(pixel, r);
                    let g = extract_channel(pixel, g);
                    let b = extract_channel(pixel, b);
                    pixels.push(Rgb::from_rgb([r, g, b]));
                }
            }
        }
        bmp::Compression::Grayscale => panic!("BMP file contains grayscale data"),
    }

    Ok(Image {
        width: bmp.width,
        height: bmp.height,
        pixels: Pixels::new(pixels),
    })
}

fn grayscale_image_from_bmp(bmp: &BmpReader) -> Result<Image<Grayscale>, &'static str> {
    let mut pixels = Vec::with_capacity((bmp.width * bmp.height) as usize);

    match bmp.compression {
        bmp::Compression::Grayscale => {
            debug_assert_eq!(bmp.bpp, 8);

            let bpr = bmp.width as usize;
            let padding = (4 - (bpr % 4)) % 4;

            let bytes = bmp.pixels;
            let input = &mut &bytes[..];
            for h in (0..bmp.height as usize).rev() {
                *input = &bytes[h * bmp.width as usize + padding..];

                for _ in 0..bmp.width {
                    let l = input[0];
                    *input = &input[1..];
                    pixels.push(l as f32 / u8::MAX as f32);
                }
            }
        }
        _ => panic!("BMP file contains RGB data"),
    }

    Ok(Image {
        width: bmp.width,
        height: bmp.height,
        pixels: Pixels::new(pixels),
    })
}
