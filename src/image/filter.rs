use std::marker::PhantomData;

use crate::image::pixel::Pixels;

use super::Image;
use super::layer::{BackPropagation, Layer};
use super::pixel::{Grayscale, Pixel};

#[derive(Debug)]
pub struct FilterData<'a, Data, Input, const WEIGHTS: usize> {
    data: Data,
    input: Option<Input>,
    filter: &'a mut Filter<WEIGHTS, Grayscale, Grayscale>,
}

impl<'a, Data, Input, const WEIGHTS: usize> FilterData<'a, Data, Input, WEIGHTS> {
    pub fn new(data: Data, filter: &'a mut Filter<WEIGHTS, Grayscale, Grayscale>) -> Self {
        Self {
            data,
            filter,
            input: None,
        }
    }
}

impl<'a, T, Input, const WEIGHTS: usize> Layer for FilterData<'a, T, Input, WEIGHTS>
where
    T: Layer<Item = Input>,
    T::Item: Filterable,
{
    type Item = Image<Grayscale>;

    fn forward(&mut self) -> Self::Item {
        let input = self.data.forward();
        self.input = Some(input.clone());
        input.filter(self.filter)
    }
}

impl<'a, T, Input, const WEIGHTS: usize> BackPropagation for FilterData<'a, T, Input, WEIGHTS>
where
    Input: Filterable,
{
    type Gradient = Image<Grayscale>;

    fn backprop(&mut self, output_gradient: Self::Gradient, learning_rate: f32) {
        let input = self
            .input
            .as_ref()
            .expect("Forward pass must be called before backprop");

        let mut filter_gradients = [0.0f32; WEIGHTS];
        let filter_size = self.filter.size as usize;
        let height = output_gradient.height() as usize;
        let width = output_gradient.width() as usize;

        for y in 0..height {
            for x in 0..width {
                let g = output_gradient.pixels[y * width + x];

                for fy in 0..filter_size {
                    for fx in 0..filter_size {
                        let weight_idx = fy * filter_size + fx;
                        if weight_idx < WEIGHTS {
                            let input_x = x + fx;
                            let input_y = y + fy;

                            if input_x < input.width() && input_y < input.height() {
                                let input_val = input.pixels()[input_y * input.width() + input_x];
                                filter_gradients[weight_idx] += input_val * g;
                            }
                        }
                    }
                }
            }
        }

        for i in 0..WEIGHTS {
            self.filter.weights[i] -= learning_rate * filter_gradients[i];
        }

        let mut bias_gradient = 0.0;
        for y in 0..height {
            for x in 0..width {
                bias_gradient += output_gradient.pixels[y * width + x];
            }
        }
        self.filter.bias -= learning_rate * bias_gradient;

        // // Step 2: Compute gradients for previous layer (input gradients)
        // // This is the "transposed convolution" operation
        // let input_height = input.height() as usize;
        // let input_width = input.width() as usize;
        //
        // // Create input gradient image (same size as input)
        // let mut input_gradients = vec![vec![0.0f32; input_width]; input_height];
        //
        // // For each input position, figure out which outputs it contributed to
        // for in_y in 0..input_height {
        //     for in_x in 0..input_width {
        //         let mut gradient_sum = 0.0;
        //
        //         // Check all filter positions and output positions
        //         for fy in 0..filter_size {
        //             for fx in 0..filter_size {
        //                 let weight_idx = fy * filter_size + fx;
        //                 if weight_idx < WEIGHTS {
        //                     // Calculate which output position this input->filter combination affects
        //                     if in_x >= fx && in_y >= fy {
        //                         let out_x = in_x - fx;
        //                         let out_y = in_y - fy;
        //
        //                         // Check if this output position exists
        //                         if out_x < output_width && out_y < output_height {
        //                             let output_grad = output_gradient
        //                                 .get_pixel(out_x as u32, out_y as u32)
        //                                 .intensity();
        //                             gradient_sum += self.filter.weights[weight_idx] * output_grad;
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //
        //         input_gradients[in_y][in_x] = gradient_sum;
        //     }
        // }
        //
        // // Step 3: Send gradients to previous layer
        // // Convert gradient matrix back to Image format and pass to previous layer
        // // (This would typically involve calling backprop on self.data with the input gradients)
        //
        // // Note: In a complete implementation, you'd need to:
        // // 1. Convert input_gradients back to Image<Grayscale> format
        // // 2. Call self.data.backprop(input_gradient_image) if T implements BackPropagation
        //
        // // For demonstration, let's show the gradient values
        // println!("Filter gradients: {:?}", filter_gradients);
        // println!("Input gradient shape: {}x{}", input_height, input_width);
        // println!("Sample input gradients (top-left 3x3):");
        // for y in 0..3.min(input_height) {
        //     for x in 0..3.min(input_width) {
        //         print!("{:6.3} ", input_gradients[y][x]);
        //     }
        //     println!();
        // }
    }
}

pub trait Filterable: Clone {
    fn width(&self) -> usize;

    fn height(&self) -> usize;

    fn pixels(&self) -> &[Grayscale];

    fn filter<const WEIGHTS: usize>(
        &self,
        filter: &Filter<WEIGHTS, Grayscale, Grayscale>,
    ) -> Image<Grayscale>;
}

impl Filterable for Image<Grayscale> {
    fn width(&self) -> usize {
        self.width as usize
    }

    fn height(&self) -> usize {
        self.height as usize
    }

    fn pixels(&self) -> &[Grayscale] {
        self.pixels.as_slice()
    }

    fn filter<const WEIGHTS: usize>(
        &self,
        filter: &Filter<WEIGHTS, Grayscale, Grayscale>,
    ) -> Image<Grayscale> {
        filter.conv_padded(self)
    }
}

impl Layer for Image<Grayscale> {
    type Item = Self;

    fn forward(&mut self) -> Self::Item {
        self.clone()
    }
}

impl Filterable for &Image<Grayscale> {
    fn width(&self) -> usize {
        self.width as usize
    }

    fn height(&self) -> usize {
        self.height as usize
    }

    fn pixels(&self) -> &[Grayscale] {
        self.pixels.as_slice()
    }

    fn filter<const WEIGHTS: usize>(
        &self,
        filter: &Filter<WEIGHTS, Grayscale, Grayscale>,
    ) -> Image<Grayscale> {
        filter.conv_padded(self)
    }
}

impl Layer for &Image<Grayscale> {
    type Item = Self;

    fn forward(&mut self) -> Self::Item {
        self
    }
}

pub fn uniform_3x3<From>() -> Filter<9, From, Grayscale>
where
    From: Pixel,
{
    Filter::new([1.0 / 3.0; 9])
}

pub fn vertical_sobel<From>() -> Filter<9, From, Grayscale>
where
    From: Pixel,
{
    #[rustfmt::skip]
    let weights = [
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ];

    Filter::new(weights)
}

pub fn horizontal_sobel<From>() -> Filter<9, From, Grayscale>
where
    From: Pixel,
{
    #[rustfmt::skip]
    let weights = [
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ];

    Filter::new(weights)
}

pub fn gaussian_blur_3x3<From>() -> Filter<9, From, From>
where
    From: Pixel,
{
    #[rustfmt::skip]
    let weights = [
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0,
    ];

    Filter::new(weights.map(|v| v / 16.0))
}

#[derive(Debug, Clone, Copy)]
pub struct Filter<const WEIGHTS: usize, From, To> {
    size: u32,
    weights: [f32; WEIGHTS],
    bias: f32,
    _from: PhantomData<From>,
    _to: PhantomData<To>,
}

impl<const WEIGHTS: usize, From, To> Filter<WEIGHTS, From, To>
where
    From: Pixel,
    To: Pixel,
{
    pub fn new(weights: [f32; WEIGHTS]) -> Self {
        assert_eq!((weights.len() as f32).sqrt().fract(), 0.0);

        Self {
            size: weights.len().isqrt() as u32,
            weights,
            bias: 0.0,
            _from: PhantomData,
            _to: PhantomData,
        }
    }

    pub fn conv(&self, image: &Image<From>) -> Image<To> {
        conv(image, self)
    }

    pub fn conv_padded(&self, image: &Image<From>) -> Image<To> {
        assert!(self.size % 2 != 0, "filter size must be odd");
        conv_padded(image, self, (self.size - 1) / 2)
    }
}

fn conv<const WEIGHTS: usize, From, To>(
    input: &Image<From>,
    filter: &Filter<WEIGHTS, From, To>,
) -> Image<To>
where
    From: Pixel,
    To: Pixel,
{
    debug_assert_eq!(input.width * input.height, input.pixels.len() as u32);
    debug_assert_eq!(filter.size * filter.size, filter.weights.len() as u32);

    debug_assert!(input.width >= filter.size);
    debug_assert!(input.height >= filter.size);

    let width = input.width - filter.size + 1;
    let height = input.height - filter.size + 1;
    let mut output = Image {
        width,
        height,
        pixels: Pixels::new(vec![To::default(); width as usize * height as usize]),
    };

    for h in 0..height as usize {
        for w in 0..width as usize {
            let mut result = [0.0; 3];
            let mut fi = 0;
            for fh in 0..filter.size as usize {
                for fw in 0..filter.size as usize {
                    let ph = fh + h;
                    let pw = fw + w;

                    let weight = filter.weights[fi];
                    let pixel = input.pixels[ph * input.width as usize + pw];
                    let rgb = pixel.to_linear_rgb();

                    result[0] += weight * rgb[0];
                    result[1] += weight * rgb[1];
                    result[2] += weight * rgb[2];

                    fi += 1;
                }
            }
            output.pixels[h * width as usize + w] =
                To::from_linear_rgb(result.map(|v| v + filter.bias));
        }
    }

    output
}

fn conv_padded<const WEIGHTS: usize, From, To>(
    input: &Image<From>,
    filter: &Filter<WEIGHTS, From, To>,
    padding: u32,
) -> Image<To>
where
    From: Pixel,
    To: Pixel,
{
    debug_assert_eq!(input.width * input.height, input.pixels.len() as u32);
    debug_assert_eq!(filter.size * filter.size, filter.weights.len() as u32);

    let width = input.width + 2 * padding - filter.size + 1;
    let height = input.height + 2 * padding - filter.size + 1;
    let mut output = Image {
        width,
        height,
        pixels: Pixels::new(vec![To::default(); width as usize * height as usize]),
    };

    let input_width = input.width as i32;
    let input_height = input.height as i32;
    let padding = padding as i32;

    for h in 0..height as usize {
        for w in 0..width as usize {
            let mut result = [0.0; 3];
            let mut fi = 0;

            for fh in 0..filter.size as usize {
                for fw in 0..filter.size as usize {
                    let img_h = h as i32 + fh as i32 - padding;
                    let img_w = w as i32 + fw as i32 - padding;

                    // TODO: remove if checks for shaders
                    let clamped_h = img_h.max(0).min(input_height - 1);
                    let clamped_w = img_w.max(0).min(input_width - 1);

                    let h_mask = !((img_h >> 31) | ((input_height - 1 - img_h) >> 31));
                    let w_mask = !((img_w >> 31) | ((input_width - 1 - img_w) >> 31));
                    let valid_mask = (h_mask & w_mask) & 1;

                    let idx = clamped_h as usize * input.width as usize + clamped_w as usize;
                    let rgb = input.pixels[idx].to_linear_rgb();
                    let weight = filter.weights[fi];
                    let mask = valid_mask as f32;

                    result[0] += weight * rgb[0] * mask;
                    result[1] += weight * rgb[1] * mask;
                    result[2] += weight * rgb[2] * mask;

                    fi += 1;
                }
            }
            output.pixels[h * width as usize + w] = To::from_linear_rgb(result);
        }
    }
    output
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn fuzz() {
        for y in 3..64 {
            for x in 3..64 {
                let image = Image {
                    width: x as u32,
                    height: y as u32,
                    pixels: Pixels::new((0..x * y).map(|p| p as f32).collect()),
                };
                let filter = vertical_sobel::<Grayscale>();
                filter.conv(&image);
            }
        }
    }

    #[test]
    fn conv_10x3_2x2() {
        let image = Image {
            width: 10,
            height: 3,
            #[rustfmt::skip]
            pixels: Pixels::new(vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2,
            ]),
        };

        let filter = Filter::<_, f32, Grayscale>::new([1.0, -1.0, -1.0, 1.0]);
        let result = filter.conv(&image);

        // Expected dimensions: (10-2+1) x (3-2+1) = 9x2
        assert_eq!(result.width, 9);
        assert_eq!(result.height, 2);
        assert_eq!(result.pixels.len(), 18);

        // Manual calculation for expected values
        let expected = vec![
            // Row 0: positions (0,0) to (8,0)
            0.1 * 1.0 + 0.2 * (-1.0) + 0.2 * (-1.0) + 0.3 * 1.0, // (0,0): 0.0
            0.2 * 1.0 + 0.3 * (-1.0) + 0.3 * (-1.0) + 0.4 * 1.0, // (1,0): 0.0
            0.3 * 1.0 + 0.4 * (-1.0) + 0.4 * (-1.0) + 0.5 * 1.0, // (2,0): 0.0
            0.4 * 1.0 + 0.5 * (-1.0) + 0.5 * (-1.0) + 0.6 * 1.0, // (3,0): 0.0
            0.5 * 1.0 + 0.6 * (-1.0) + 0.6 * (-1.0) + 0.7 * 1.0, // (4,0): 0.0
            0.6 * 1.0 + 0.7 * (-1.0) + 0.7 * (-1.0) + 0.8 * 1.0, // (5,0): 0.0
            0.7 * 1.0 + 0.8 * (-1.0) + 0.8 * (-1.0) + 0.9 * 1.0, // (6,0): 0.0
            0.8 * 1.0 + 0.9 * (-1.0) + 0.9 * (-1.0) + 1.0 * 1.0, // (7,0): 0.0
            0.9 * 1.0 + 1.0 * (-1.0) + 1.0 * (-1.0) + 0.1 * 1.0, // (8,0): -1.0
            // Row 1: positions (0,1) to (8,1)
            0.2 * 1.0 + 0.3 * (-1.0) + 0.3 * (-1.0) + 0.4 * 1.0, // (0,1): 0.0
            0.3 * 1.0 + 0.4 * (-1.0) + 0.4 * (-1.0) + 0.5 * 1.0, // (1,1): 0.0
            0.4 * 1.0 + 0.5 * (-1.0) + 0.5 * (-1.0) + 0.6 * 1.0, // (2,1): 0.0
            0.5 * 1.0 + 0.6 * (-1.0) + 0.6 * (-1.0) + 0.7 * 1.0, // (3,1): 0.0
            0.6 * 1.0 + 0.7 * (-1.0) + 0.7 * (-1.0) + 0.8 * 1.0, // (4,1): 0.0
            0.7 * 1.0 + 0.8 * (-1.0) + 0.8 * (-1.0) + 0.9 * 1.0, // (5,1): 0.0
            0.8 * 1.0 + 0.9 * (-1.0) + 0.9 * (-1.0) + 1.0 * 1.0, // (6,1): 0.0
            0.9 * 1.0 + 1.0 * (-1.0) + 1.0 * (-1.0) + 0.1 * 1.0, // (7,1): -1.0
            1.0 * 1.0 + 0.1 * (-1.0) + 0.1 * (-1.0) + 0.2 * 1.0, // (8,1): 1.0
        ];

        for (i, (&actual, &expected)) in result.pixels.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Test 1: Pixel {} mismatch: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn conv_4x4_1x1() {
        let image = Image {
            width: 4,
            height: 4,
            #[rustfmt::skip]
            pixels: Pixels::new(vec![
                0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 0.0, 0.1,
                0.2, 0.3, 0.4, 0.5,
            ]),
        };

        let filter = Filter::<_, f32, Grayscale>::new([2.0]);
        let result = filter.conv(&image);

        // Expected dimensions: (4-1+1) x (4-1+1) = 4x4
        assert_eq!(result.width, 4);
        assert_eq!(result.height, 4);
        assert_eq!(result.pixels.len(), 16);

        // Manual calculation: each pixel multiplied by 2.0
        let expected: Vec<f32> = image.pixels.iter().map(|&p| p * 2.0).collect();

        for (i, (&actual, &expected)) in result.pixels.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Test 2: Pixel {} mismatch: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn conv_7x7_3x3() {
        let image = Image {
            width: 7,
            height: 7,
            #[rustfmt::skip]
            pixels: Pixels::new(vec![
                0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0,
                0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1,
            ]),
        };

        // Use a simple 3x3 box filter (all weights = 1/9)
        let filter = Filter::<_, f32, Grayscale>::new([1.0 / 9.0; 9]);
        let result = filter.conv(&image);

        // Expected dimensions: (7-3+1) x (7-3+1) = 5x5
        assert_eq!(result.width, 5);
        assert_eq!(result.height, 5);
        assert_eq!(result.pixels.len(), 25);

        // Manual calculation for a few key positions
        // Position (0,0) - top-left 3x3 region average
        let expected_0_0 = (0.0 + 0.1 + 0.2 + 0.1 + 0.2 + 0.3 + 0.2 + 0.3 + 0.4) / 9.0;
        assert!(
            (result.pixels[0] - expected_0_0).abs() < 1e-6,
            "Test 3: Position (0,0) mismatch: expected {}, got {}",
            expected_0_0,
            result.pixels[0]
        );

        // Position (2,2) - center 3x3 region average
        let expected_2_2 = (0.4 + 0.5 + 0.6 + 0.5 + 0.6 + 0.7 + 0.6 + 0.7 + 0.8) / 9.0;
        assert!(
            (result.pixels[2 * 5 + 2] - expected_2_2).abs() < 1e-6,
            "Test 3: Position (2,2) mismatch: expected {}, got {}",
            expected_2_2,
            result.pixels[2 * 5 + 2]
        );

        // Position (4,4) - bottom-right 3x3 region average
        let expected_4_4 = (0.8 + 0.9 + 1.0 + 0.9 + 1.0 + 0.0 + 1.0 + 0.0 + 0.1) / 9.0;
        assert!(
            (result.pixels[4 * 5 + 4] - expected_4_4).abs() < 1e-6,
            "Test 3: Position (4,4) mismatch: expected {}, got {}",
            expected_4_4,
            result.pixels[4 * 5 + 4]
        );

        // Verify all pixels are within expected range [0, 1]
        for (i, &pixel) in result.pixels.iter().enumerate() {
            assert!(
                pixel >= 0.0 && pixel <= 1.0,
                "Test 3: Pixel {} out of range [0,1]: {}",
                i,
                pixel
            );
        }
    }

    #[test]
    fn conv_padded() {
        let padded_image = Image {
            width: 7,
            height: 7,
            #[rustfmt::skip]
            pixels: Pixels::new(vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0,
                0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0,
                0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0,
                0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0,
                0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
        };
        let normal_image = Image {
            width: 5,
            height: 5,
            #[rustfmt::skip]
            pixels: Pixels::new(vec![
                0.2, 0.3, 0.4, 0.5, 0.6,
                0.3, 0.4, 0.5, 0.6, 0.7,
                0.4, 0.5, 0.6, 0.7, 0.8,
                0.5, 0.6, 0.7, 0.8, 0.9,
                0.6, 0.7, 0.8, 0.9, 1.0,
            ]),
        };

        let filter = Filter::<_, f32, Grayscale>::new([1.0 / 9.0; 9]);

        let conv_result = filter.conv(&padded_image);
        let conv_padded_result = filter.conv_padded(&normal_image);

        assert_eq!(
            conv_result.pixels.as_slice(),
            conv_padded_result.pixels.as_slice()
        );
        assert_eq!(conv_result.width, conv_padded_result.width);
        assert_eq!(conv_result.height, conv_padded_result.height);
    }
}
