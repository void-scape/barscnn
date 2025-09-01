use crate::image::Image;
use crate::layer::{BackPropagation, Layer};

#[derive(Debug, Clone)]
pub struct MaxPool<Data> {
    pub data: Data,
    pub size: usize,
    pub input: Image,
}

pub trait MaxPoolData
where
    Self: Sized,
{
    fn max_pool(self, size: usize) -> MaxPool<Self>;
}

impl<T> MaxPoolData for T
where
    T: Layer<Item = Image>,
{
    fn max_pool(self, size: usize) -> MaxPool<Self> {
        MaxPool {
            data: self,
            size,
            input: Image::default(),
        }
    }
}

impl<T> Layer for MaxPool<T>
where
    T: Layer<Item = Image>,
{
    type Input = T::Input;
    type Item = T::Item;

    fn forward(&mut self, input: Self::Input) -> Self::Item {
        let input = self.data.forward(input);
        let result = max_pool(&input, self.size);
        self.input = input;
        result
    }
}

impl<T> BackPropagation for MaxPool<T>
where
    T: BackPropagation<Gradient = Image>,
{
    type Gradient = Image;

    fn backprop(&mut self, output_gradient: Self::Gradient) {
        self.data
            .backprop(max_pool_reshape(&self.input, &output_gradient, self.size));
    }

    fn learning_rate(&self) -> f32 {
        self.data.learning_rate()
    }
}

pub fn max_pool(input: &Image, size: usize) -> Image {
    assert!(size != 0);

    debug_assert_eq!(
        input.width * input.height * input.channels,
        input.pixels.len()
    );
    debug_assert!(size <= input.width && size <= input.height);

    let width = input.width / size;
    let height = input.height / size;
    let channels = input.channels;
    let mut output = Image {
        width,
        height,
        channels,
        pixels: vec![0.0; width * height * channels],
    };

    for mh in 0..height {
        for mw in 0..width {
            for c in 0..channels {
                let mut max = f32::MIN;
                for h in 0..size {
                    for w in 0..size {
                        let pixel = input.pixels[((mh * size + h) * input.width + (mw * size + w))
                            * input.channels
                            + c];
                        max = pixel.max(max);
                    }
                }
                output.pixels[(mh * width + mw) * channels + c] = max;
            }
        }
    }

    output
}

fn max_pool_reshape(original: &Image, pooled: &Image, size: usize) -> Image {
    assert!(size != 0);
    assert_eq!(original.channels, pooled.channels);

    debug_assert_eq!(
        original.width * original.height * original.channels,
        original.pixels.len()
    );
    debug_assert_eq!(
        pooled.width * pooled.height * pooled.channels,
        pooled.pixels.len()
    );
    debug_assert_eq!(pooled.width, original.width / size);
    debug_assert!(size <= original.width && size <= original.height);

    let mut output = Image {
        width: original.width,
        height: original.height,
        channels: original.channels,
        pixels: vec![0.0; original.width * original.height * original.channels],
    };

    for mh in 0..pooled.height {
        for mw in 0..pooled.width {
            for c in 0..pooled.channels {
                let mut max = f32::MIN;
                let mut i = 0;
                for h in 0..size {
                    for w in 0..size {
                        let index = ((mh * size + h) * original.width + (mw * size + w))
                            * original.channels
                            + c;
                        let pixel = original.pixels[index];
                        if pixel > max {
                            max = pixel;
                            i = index;
                        }
                    }
                }
                output.pixels[i] = pooled.pixels[(mh * pooled.width + mw) * original.channels + c];
            }
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
                        width: x,
                        height: y,
                        channels: 1,
                        pixels: (0..x * y).map(|p| p as f32).collect(),
                    };
                    max_pool(&image, pool);
                }
            }
        }
    }

    #[test]
    fn max_pool_5x3_2() {
        let image = Image {
            width: 5,
            height: 3,
            channels: 1,
            #[rustfmt::skip]
            pixels: vec![
                0.1, 0.2, 0.3, 0.4, 0.5,
                0.2, 0.3, 0.4, 0.5, 0.6,
                0.3, 0.4, 0.5, 0.6, 0.7,
            ],
        };

        let result = max_pool(&image, 2);

        assert_eq!(result.width, 2);
        assert_eq!(result.height, 1);
        assert_eq!(result.pixels.len(), 2);
        assert_eq!(result.pixels.as_slice(), &[0.3, 0.5]);

        let unmax = max_pool_reshape(&image, &result, 2);

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
