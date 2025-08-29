use crate::image::pixel::Pixels;

use super::Image;
use super::pixel::{Grayscale, Pixel};

pub fn max_pool<T>(pool: usize, input: &Image<T>) -> Image<Grayscale>
where
    T: Pixel,
{
    assert!(pool != 0);
    debug_assert_eq!(input.width * input.height, input.pixels.len() as u32);
    debug_assert!(pool as u32 <= input.width && pool as u32 <= input.height);
    debug_assert!(pool != 0);

    let width = input.width / pool as u32;
    let height = input.height / pool as u32;
    let mut output = Image {
        width,
        height,
        pixels: Pixels::new(vec![Grayscale::default(); width as usize * height as usize]),
    };

    let size = pool;
    let mut i = 0;
    for h in 0..height as usize {
        for w in 0..width as usize {
            let h = h * size;
            let w = w * size;

            let max = (0..size)
                .map(|ph| {
                    (0..size).map(move |pw| {
                        input.pixels[(h + ph) * input.width as usize + w + pw].luminance()
                    })
                })
                .flatten()
                // TODO: Remove if checks for shaders.
                .max_by(|a, b| a.total_cmp(b))
                .unwrap();

            output.pixels[i] = max;
            i += 1;
        }
    }

    output
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn fuzz() {
        for pool in 1..5 {
            for y in 5..64 {
                for x in 5..64 {
                    let image = Image {
                        width: x as u32,
                        height: y as u32,
                        pixels: Pixels::new((0..x * y).map(|p| p as f32).collect()),
                    };
                    max_pool(pool, &image);
                }
            }
        }
    }

    #[test]
    fn max_pool_5x3_2() {
        let image = Image {
            width: 5,
            height: 3,
            #[rustfmt::skip]
            pixels: Pixels::new(vec![
                0.1, 0.2, 0.3, 0.4, 0.5,
                0.2, 0.3, 0.4, 0.5, 0.6,
                0.3, 0.4, 0.5, 0.6, 0.7,
            ]),
        };

        let result = max_pool(2, &image);

        assert_eq!(result.width, 2);
        assert_eq!(result.height, 1);
        assert_eq!(result.pixels.len(), 2);
        assert_eq!(result.pixels.as_slice(), &[0.3, 0.5]);
    }
}
