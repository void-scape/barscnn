use std::marker::PhantomData;

use crate::image::pixel::Pixels;

use super::Image;
use super::feature::FeatureSet;
use super::pixel::{Grayscale, Pixel};

pub fn identity<const DEPTH: usize, From, To>() -> Filter<9, DEPTH, From, To>
where
    From: Pixel,
    To: Pixel,
{
    #[rustfmt::skip]
    let weights = [
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ];

    Filter::new([weights; DEPTH])
}

pub fn uniform_3x3<const DEPTH: usize, From>() -> Filter<9, DEPTH, From, Grayscale>
where
    From: Pixel,
{
    Filter::new([[1.0 / 6.0; 9]; DEPTH])
}

pub fn vertical_sobel<const DEPTH: usize, From>() -> Filter<9, DEPTH, From, Grayscale>
where
    From: Pixel,
{
    #[rustfmt::skip]
    let weights = [
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ];

    Filter::new([weights; DEPTH])
}

pub fn horizontal_sobel<const DEPTH: usize, From>() -> Filter<9, DEPTH, From, Grayscale>
where
    From: Pixel,
{
    #[rustfmt::skip]
    let weights = [
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ];

    Filter::new([weights; DEPTH])
}

pub fn gaussian_blur_3x3<const DEPTH: usize, From>() -> Filter<9, DEPTH, From, From>
where
    From: Pixel,
{
    #[rustfmt::skip]
    let weights = [
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0,
    ];

    Filter::new([weights.map(|v| v / 16.0); DEPTH])
}

pub fn laplacian_3x3<const DEPTH: usize, From>() -> Filter<9, DEPTH, From, Grayscale>
where
    From: Pixel,
{
    #[rustfmt::skip]
    let weights = [
         0.0, -1.0,  0.0,
        -1.0,  4.0, -1.0,
         0.0, -1.0,  0.0,
    ];

    Filter::new([weights; DEPTH])
}

#[derive(Debug, Clone, Copy)]
pub struct Filter<const WEIGHTS: usize, const DEPTH: usize, From, To> {
    size: u32,
    weights: [[f32; WEIGHTS]; DEPTH],
    bias: f32,
    _from: PhantomData<From>,
    _to: PhantomData<To>,
}

impl<const WEIGHTS: usize, const DEPTH: usize, From, To> Filter<WEIGHTS, DEPTH, From, To>
where
    From: Pixel,
    To: Pixel,
{
    pub fn new(weights: [[f32; WEIGHTS]; DEPTH]) -> Self {
        assert_eq!((WEIGHTS as f32).sqrt().fract(), 0.0);

        Self {
            size: (WEIGHTS as f32).sqrt() as u32,
            weights,
            bias: 0.0,
            _from: PhantomData,
            _to: PhantomData,
        }
    }

    pub fn weights(&self) -> &[[f32; WEIGHTS]; DEPTH] {
        &self.weights
    }
}

impl<const WEIGHTS: usize, const DEPTH: usize> Filter<WEIGHTS, DEPTH, Grayscale, Grayscale> {
    pub fn compute_input_gradients<T>(
        &self,
        input: &T,
        output_gradient: &Image<Grayscale>,
    ) -> T::Gradient
    where
        T: Filterable<WEIGHTS, DEPTH> + GradientReceiver<WEIGHTS, DEPTH>,
    {
        let mut input_gradient = input.zero_gradient();
        input.accumulate_gradient(&mut input_gradient, output_gradient, self);
        input_gradient
    }

    pub fn backprop<T>(&mut self, learning_rate: f32, input: &T, output_gradient: &Image<Grayscale>)
    where
        T: Filterable<WEIGHTS, DEPTH>,
    {
        for depth in 0..DEPTH {
            let mut filter_gradients = [0.0f32; WEIGHTS];
            let filter_size = self.size as usize;
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
                                    let input_val =
                                        input.pixels(depth)[input_y * input.width() + input_x];
                                    filter_gradients[weight_idx] += input_val * g;
                                }
                            }
                        }
                    }
                }
            }

            for i in 0..WEIGHTS {
                self.weights[depth][i] -= learning_rate * filter_gradients[i];
            }

            let mut bias_gradient = 0.0;
            for y in 0..height {
                for x in 0..width {
                    bias_gradient += output_gradient.pixels[y * width + x];
                }
            }
            self.bias -= learning_rate * bias_gradient;
        }
    }
}

pub trait Filterable<const WEIGHTS: usize, const DEPTH: usize>: Clone {
    fn width(&self) -> usize;

    fn height(&self) -> usize;

    fn pixels(&self, depth: usize) -> &[Grayscale];

    fn filter(&self, filter: &Filter<WEIGHTS, DEPTH, Grayscale, Grayscale>) -> Image<Grayscale>;
}

impl<const WEIGHTS: usize> Filterable<WEIGHTS, 1> for Image<Grayscale> {
    fn width(&self) -> usize {
        self.width as usize
    }

    fn height(&self) -> usize {
        self.height as usize
    }

    fn pixels(&self, depth: usize) -> &[Grayscale] {
        assert!(depth == 0);
        self.pixels.as_slice()
    }

    fn filter(&self, filter: &Filter<WEIGHTS, 1, Grayscale, Grayscale>) -> Image<Grayscale> {
        assert!(filter.size % 2 != 0, "filter size must be odd");
        conv_padded(&[self], filter, (filter.size - 1) / 2)
    }
}

impl<const WEIGHTS: usize, const DEPTH: usize> Filterable<WEIGHTS, DEPTH> for FeatureSet<DEPTH> {
    fn width(&self) -> usize {
        debug_assert!(
            self.iter()
                .all(|img| img.width() == self[0].width() && img.height() == self[0].height())
        );
        self[0].width()
    }

    fn height(&self) -> usize {
        debug_assert!(
            self.iter()
                .all(|img| img.width() == self[0].width() && img.height() == self[0].height())
        );
        self[0].height()
    }

    fn pixels(&self, depth: usize) -> &[Grayscale] {
        self[depth].pixels.as_slice()
    }

    fn filter(&self, filter: &Filter<WEIGHTS, DEPTH, Grayscale, Grayscale>) -> Image<Grayscale> {
        assert!(filter.size % 2 != 0, "filter size must be odd");
        conv_padded(&self.each_ref(), filter, (filter.size - 1) / 2)
    }
}

#[allow(unused)]
fn conv<const WEIGHTS: usize, const DEPTH: usize, From, To>(
    input: &[&Image<From>; DEPTH],
    filter: &Filter<WEIGHTS, DEPTH, From, To>,
) -> Image<To>
where
    From: Pixel,
    To: Pixel,
{
    debug_assert!(
        input
            .iter()
            .all(|img| img.width() == input[0].width() && img.height() == input[0].height())
    );

    let width = input[0].width() as u32 - filter.size + 1;
    let height = input[0].height() as u32 - filter.size + 1;
    let mut output = Image {
        width,
        height,
        pixels: Pixels::new(vec![To::default(); width as usize * height as usize]),
    };

    for (i, input) in input.iter().enumerate() {
        debug_assert_eq!(input.width() * input.height(), input.pixels.len());
        debug_assert_eq!(filter.size * filter.size, filter.weights[i].len() as u32);

        debug_assert!(input.width() as u32 >= filter.size);
        debug_assert!(input.height() as u32 >= filter.size);

        for h in 0..height as usize {
            for w in 0..width as usize {
                let mut result = [0.0; 3];
                let mut fi = 0;
                for fh in 0..filter.size as usize {
                    for fw in 0..filter.size as usize {
                        let ph = fh + h;
                        let pw = fw + w;

                        let weight = filter.weights[i][fi];
                        let pixel = input.pixels[ph * input.width() + pw];
                        let rgb = pixel.to_linear_rgb();

                        result[0] += weight * rgb[0];
                        result[1] += weight * rgb[1];
                        result[2] += weight * rgb[2];

                        fi += 1;
                    }
                }
                let current = output.pixels[h * width as usize + w].to_linear_rgb();
                let mut result = result.map(|v| v + filter.bias);
                for i in 0..3 {
                    result[i] += current[i];
                }
                output.pixels[h * width as usize + w] = To::from_linear_rgb(result);
            }
        }
    }

    output
        .pixels
        .iter_mut()
        .for_each(|p| *p = To::from_linear_rgb(p.to_linear_rgb().map(|v| v + filter.bias)));

    output
}

fn conv_padded<const WEIGHTS: usize, const DEPTH: usize, From, To>(
    input: &[&Image<From>; DEPTH],
    filter: &Filter<WEIGHTS, DEPTH, From, To>,
    padding: u32,
) -> Image<To>
where
    From: Pixel,
    To: Pixel,
{
    debug_assert!(
        input
            .iter()
            .all(|img| img.width() == input[0].width() && img.height() == input[0].height())
    );

    let width = input[0].width() as u32 + 2 * padding - filter.size + 1;
    let height = input[0].height() as u32 + 2 * padding - filter.size + 1;
    let mut output = Image {
        width,
        height,
        pixels: Pixels::new(vec![To::default(); width as usize * height as usize]),
    };

    for (i, input) in input.iter().enumerate() {
        debug_assert_eq!(input.width() * input.height(), input.pixels.len());
        debug_assert_eq!(filter.size * filter.size, filter.weights[i].len() as u32);

        let input_width = input.width() as i32;
        let input_height = input.height() as i32;
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

                        let idx = clamped_h as usize * input.width() + clamped_w as usize;
                        let rgb = input.pixels[idx].to_linear_rgb();
                        let weight = filter.weights[i][fi];
                        let mask = valid_mask as f32;

                        result[0] += weight * rgb[0] * mask;
                        result[1] += weight * rgb[1] * mask;
                        result[2] += weight * rgb[2] * mask;

                        fi += 1;
                    }
                }
                let current = output.pixels[h * width as usize + w].to_linear_rgb();
                for i in 0..3 {
                    result[i] += current[i];
                }
                output.pixels[h * width as usize + w] = To::from_linear_rgb(result);
            }
        }
    }

    output
        .pixels
        .iter_mut()
        .for_each(|p| *p = To::from_linear_rgb(p.to_linear_rgb().map(|v| v + filter.bias)));

    output
}

pub trait GradientReceiver<const WEIGHTS: usize, const DEPTH: usize> {
    type Gradient;

    fn zero_gradient(&self) -> Self::Gradient;

    fn accumulate_gradient(
        &self,
        gradient: &mut Self::Gradient,
        filter_gradient: &Image<Grayscale>,
        filter: &Filter<WEIGHTS, DEPTH, Grayscale, Grayscale>,
    );
}

impl<const WEIGHTS: usize, const DEPTH: usize> GradientReceiver<WEIGHTS, DEPTH>
    for Image<Grayscale>
{
    type Gradient = Image<Grayscale>;

    fn zero_gradient(&self) -> Self::Gradient {
        Image {
            width: self.width,
            height: self.height,
            pixels: Pixels::new(vec![0.0; (self.width * self.height) as usize]),
        }
    }

    fn accumulate_gradient(
        &self,
        input_gradient: &mut Self::Gradient,
        output_gradient: &Image<Grayscale>,
        filter: &Filter<WEIGHTS, DEPTH, Grayscale, Grayscale>,
    ) {
        let filter_size = (WEIGHTS as f32).sqrt() as usize;
        let input_height = self.height as usize;
        let input_width = self.width as usize;
        let output_height = output_gradient.height() as usize;
        let output_width = output_gradient.width() as usize;

        for input_y in 0..input_height {
            for input_x in 0..input_width {
                let mut gradient_sum = 0.0;

                for fy in 0..filter_size {
                    for fx in 0..filter_size {
                        let weight_idx = fy * filter_size + fx;
                        if weight_idx < WEIGHTS {
                            if input_x >= fx && input_y >= fy {
                                let output_x = input_x - fx;
                                let output_y = input_y - fy;

                                if output_x < output_width && output_y < output_height {
                                    let output_grad =
                                        output_gradient.pixels[output_y * output_width + output_x];
                                    let weight = filter.weights()[0][weight_idx];
                                    gradient_sum += weight * output_grad;
                                }
                            }
                        }
                    }
                }

                input_gradient.pixels[input_y * input_width + input_x] += gradient_sum;
            }
        }
    }
}

impl<const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize> GradientReceiver<WEIGHTS, DEPTH>
    for FeatureSet<WIDTH>
{
    type Gradient = FeatureSet<WIDTH>;

    fn zero_gradient(&self) -> Self::Gradient {
        let zero_features: [Image<Grayscale>; WIDTH] = std::array::from_fn(|i| {
            let feature = &self[i];
            Image {
                width: feature.width,
                height: feature.height,
                pixels: Pixels::new(vec![0.0; (feature.width * feature.height) as usize]),
            }
        });
        FeatureSet::new(zero_features)
    }

    fn accumulate_gradient(
        &self,
        input_gradient: &mut Self::Gradient,
        output_gradient: &Image<Grayscale>,
        filter: &Filter<WEIGHTS, DEPTH, Grayscale, Grayscale>,
    ) {
        let filter_size = filter.size as usize;
        for feature_idx in 0..WIDTH {
            let input_feature = &self[feature_idx];
            let gradient_feature = &mut input_gradient[feature_idx];

            let input_height = input_feature.height as usize;
            let input_width = input_feature.width as usize;
            let output_height = output_gradient.height() as usize;
            let output_width = output_gradient.width() as usize;

            for input_y in 0..input_height {
                for input_x in 0..input_width {
                    let mut gradient_sum = 0.0;

                    for fy in 0..filter_size {
                        for fx in 0..filter_size {
                            let weight_idx = fy * filter_size + fx;
                            if weight_idx < WEIGHTS {
                                if input_x >= fx && input_y >= fy {
                                    let output_x = input_x - fx;
                                    let output_y = input_y - fy;

                                    if output_x < output_width && output_y < output_height {
                                        let output_grad = output_gradient.pixels
                                            [output_y * output_width + output_x];
                                        let weight = filter.weights()[feature_idx][weight_idx];
                                        gradient_sum += weight * output_grad;
                                    }
                                }
                            }
                        }
                    }

                    gradient_feature.pixels[input_y * input_width + input_x] += gradient_sum;
                }
            }
        }
    }
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
                let filter = vertical_sobel::<_, Grayscale>();
                conv(&[&image], &filter);
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

        let filter = Filter::<_, _, f32, Grayscale>::new([[1.0, -1.0, -1.0, 1.0]]);
        let result = conv(&[&image], &filter);

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

        let filter = Filter::<_, _, f32, Grayscale>::new([[2.0]]);
        let result = conv(&[&image], &filter);

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
        let filter = Filter::<_, _, f32, Grayscale>::new([[1.0 / 9.0; 9]]);
        let result = conv(&[&image], &filter);

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

        let filter = Filter::<_, _, f32, Grayscale>::new([[1.0 / 9.0; 9]]);

        let conv_result = conv(&[&padded_image], &filter);
        let conv_padded_result = normal_image.filter(&filter);

        assert_eq!(
            conv_result.pixels.as_slice(),
            conv_padded_result.pixels.as_slice()
        );
        assert_eq!(conv_result.width, conv_padded_result.width);
        assert_eq!(conv_result.height, conv_padded_result.height);
    }

    #[test]
    fn conv_features() {
        let image1 = Image {
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

        let image2 = Image {
            width: 4,
            height: 4,
            #[rustfmt::skip]
            pixels: Pixels::new(vec![
                0.1, 0.1, 0.3, 0.4,
                0.1, 0.6, 2.7, 0.8,
                0.9, 1.9, 0.3, 0.1,
                0.2, 0.0, 0.4, 0.5,
            ]),
        };

        let f1 = horizontal_sobel();
        let r1 = conv(&[&image1], &f1);
        let r2 = conv(&[&image2], &f1);
        let result = r1
            .pixels
            .iter()
            .zip(r2.pixels.iter())
            .map(|(p1, p2)| *p1 + *p2)
            .collect::<Vec<_>>();

        let f1 = horizontal_sobel();
        let r3 = conv(&[&image1, &image2], &f1);

        assert_eq!(result.len(), r3.pixels.len());
        for (p1, p2) in result.iter().zip(r3.pixels.iter()) {
            assert!((*p1 - *p2).abs() < f32::EPSILON * 6.0);
        }
    }
}
