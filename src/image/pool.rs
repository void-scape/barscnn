use crate::image::pixel::Pixels;

use super::Image;
use super::feature::FeatureSet;
use super::layer::{BackPropagation, Layer};
use super::pixel::{Grayscale, Pixel};

#[derive(Debug)]
pub struct MaxPool<Data, Input> {
    size: usize,
    data: Data,
    input: Input,
}

pub trait MaxPoolData<Input>
where
    Self: Sized,
{
    fn max_pool(self, size: usize) -> MaxPool<Self, Input>;
}

impl<'a, T, Input> MaxPoolData<Input> for T
where
    T: Layer<'a, Item = Input>,
    T::Item: MaxPoolable,
{
    fn max_pool(self, size: usize) -> MaxPool<Self, T::Item> {
        MaxPool {
            size,
            data: self,
            input: T::Item::default(),
        }
    }
}

impl<'a, T, Input> Layer<'a> for MaxPool<T, Input>
where
    T: Layer<'a, Item = Input>,
    T::Item: MaxPoolable,
{
    type Input = T::Input;
    type Item = T::Item;

    fn forward(&mut self, input: Self::Input) -> Self::Item {
        let input = self.data.forward(input);
        self.input = input.clone();
        input.max_pool(self.size)
    }
}

impl<T, Input> BackPropagation for MaxPool<T, Input>
where
    T: BackPropagation<Gradient = Input>,
    Input: MaxPoolable,
{
    type Gradient = Input;

    fn backprop(&mut self, output_gradient: Self::Gradient) {
        let unmaxed = <Input as MaxPoolable>::unmax_pool(&self.input, output_gradient, self.size);
        self.data.backprop(unmaxed);
    }

    fn learning_rate(&self) -> f32 {
        self.data.learning_rate()
    }
}

trait MaxPoolable: Default + Clone {
    fn max_pool(self, size: usize) -> Self;
    fn unmax_pool(original: &Self, pooled: Self, size: usize) -> Self;
}

impl MaxPoolable for Image<Grayscale> {
    fn max_pool(self, size: usize) -> Self {
        max_pool(size, &self)
    }

    fn unmax_pool(original: &Self, pooled: Self, size: usize) -> Self {
        unmax_pool(original, &pooled, size)
    }
}

impl<const DEPTH: usize> MaxPoolable for FeatureSet<DEPTH> {
    fn max_pool(self, size: usize) -> Self {
        FeatureSet::new(self.each_ref().map(|img| max_pool(size, img)))
    }

    fn unmax_pool(original: &Self, pooled: Self, size: usize) -> Self {
        Self::new(std::array::from_fn(|i| {
            unmax_pool(&original[i], &pooled[i], size)
        }))
    }
}

fn max_pool<T>(pool: usize, input: &Image<T>) -> Image<Grayscale>
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

fn unmax_pool(original: &Image<f32>, pooled: &Image<f32>, size: usize) -> Image<f32> {
    let width = original.width / size as u32;
    let height = original.height / size as u32;
    let mut output = Image {
        width: original.width,
        height: original.height,
        pixels: Pixels::new(vec![
            Grayscale::default();
            original.width as usize * original.height as usize
        ]),
    };

    let pxls = original.pixels.as_slice();
    for ph in 0..height as usize {
        for pw in 0..width as usize {
            let h = ph * size;
            let w = pw * size;

            let max = (0..size)
                .map(|ph| {
                    (0..size).map(move |pw| {
                        let index = (h + ph) * original.width as usize + w + pw;
                        (index, pxls[index])
                    })
                })
                .flatten()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap();

            output.pixels[max] = pooled.pixels[ph * width as usize + pw];
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

        let unmax = unmax_pool(&image, &result, 2);

        #[rustfmt::skip]
        assert_eq!(
            unmax.pixels.as_slice(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.3, 0.0, 0.5, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );
    }
}
