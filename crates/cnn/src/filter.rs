use crate::image::Image;

pub fn identity<const DEPTH: usize>() -> Filter<9, DEPTH> {
    #[rustfmt::skip]
    let weights = [
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ];

    Filter::new([weights; DEPTH])
}

pub fn uniform_3x3<const DEPTH: usize>() -> Filter<9, DEPTH> {
    Filter::new([[1.0 / 6.0; 9]; DEPTH])
}

pub fn vertical_sobel<const DEPTH: usize>() -> Filter<9, DEPTH> {
    #[rustfmt::skip]
    let weights = [
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ];

    Filter::new([weights; DEPTH])
}

pub fn horizontal_sobel<const DEPTH: usize>() -> Filter<9, DEPTH> {
    #[rustfmt::skip]
    let weights = [
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ];

    Filter::new([weights; DEPTH])
}

pub fn gaussian_blur_3x3<const DEPTH: usize>() -> Filter<9, DEPTH> {
    #[rustfmt::skip]
    let weights = [
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0,
    ];

    Filter::new([weights.map(|v| v / 16.0); DEPTH])
}

pub fn laplacian_3x3<const DEPTH: usize>() -> Filter<9, DEPTH> {
    #[rustfmt::skip]
    let weights = [
         0.0, -1.0,  0.0,
        -1.0,  4.0, -1.0,
         0.0, -1.0,  0.0,
    ];

    Filter::new([weights; DEPTH])
}

#[derive(Debug, Clone, Copy)]
pub struct Filter<const WEIGHTS: usize, const DEPTH: usize> {
    pub size: usize,
    pub weights: [[f32; WEIGHTS]; DEPTH],
    pub bias: f32,
}

impl<const WEIGHTS: usize, const DEPTH: usize> Filter<WEIGHTS, DEPTH> {
    pub fn new(weights: [[f32; WEIGHTS]; DEPTH]) -> Self {
        assert_eq!((WEIGHTS as f32).sqrt().fract(), 0.0);
        Self {
            size: (WEIGHTS as f32).sqrt() as usize,
            weights,
            bias: 0.0,
        }
    }

    pub fn weights(&self) -> &[[f32; WEIGHTS]; DEPTH] {
        &self.weights
    }

    pub fn apply_gradients(
        &mut self,
        input: &Image,
        gradients: &Image,
        input_gradients: &mut Image,
        channel_index: usize,
        learning_rate: f32,
    ) {
        debug_assert_eq!(
            input.width * input.height * input.channels,
            input.pixels.len()
        );
        debug_assert_eq!(
            gradients.width * gradients.height * gradients.channels,
            gradients.pixels.len()
        );
        debug_assert_eq!(
            input_gradients.width * input_gradients.height * input_gradients.channels,
            input_gradients.pixels.len()
        );
        debug_assert_eq!(input.channels, DEPTH);
        debug_assert_eq!(input.shape(), input_gradients.shape());

        debug_assert_eq!(input.width, input_gradients.width);
        debug_assert_eq!(input.height, input_gradients.height);

        debug_assert_eq!(input.width, gradients.width);
        debug_assert_eq!(input.height, gradients.height);

        let mut filter_gradients = [[0.0; WEIGHTS]; DEPTH];

        for y in 0..gradients.height.min(input.height - self.size + 1) {
            for x in 0..gradients.width.min(input.width - self.size + 1) {
                let g = gradients.pixels
                    [(y * gradients.width + x) * gradients.channels + channel_index];

                for c in 0..DEPTH {
                    for fy in 0..self.size {
                        for fx in 0..self.size {
                            let index = ((y + fy) * input.width + x + fx) * input.channels + c;

                            filter_gradients[c][fy * self.size + fx] += input.pixels[index] * g;

                            input_gradients.pixels[index] +=
                                g * self.weights[c][fy * self.size + fx];
                        }
                    }
                }
            }
        }

        for c in 0..DEPTH {
            for w in 0..WEIGHTS {
                self.weights[c][w] -= learning_rate * filter_gradients[c][w];
            }
        }
    }
}

#[allow(unused)]
pub fn conv<const WEIGHTS: usize, const DEPTH: usize>(
    input: &Image,
    filters: &[Filter<WEIGHTS, DEPTH>],
) -> Image {
    assert_eq!(input.channels, DEPTH);

    let filter_size = filters[0].size;
    let out_width = input.width - filter_size + 1;
    let out_height = input.height - filter_size + 1;

    let mut output = Image {
        width: out_width,
        height: out_height,
        channels: filters.len(),
        pixels: vec![0.0; out_width * out_height * filters.len()],
    };

    for (filter_idx, filter) in filters.iter().enumerate() {
        for out_y in 0..out_height {
            for out_x in 0..out_width {
                let mut result = 0.0;

                for fy in 0..filter_size {
                    for fx in 0..filter_size {
                        for c in 0..DEPTH {
                            let in_y = out_y + fy;
                            let in_x = out_x + fx;
                            let input_idx = (in_y * input.width + in_x) * input.channels + c;
                            let filter_idx_2d = fy * filter_size + fx;

                            result += input.pixels[input_idx] * filter.weights[c][filter_idx_2d];
                        }
                    }
                }

                let output_idx = (out_y * out_width + out_x) * filters.len() + filter_idx;
                output.pixels[output_idx] = result + filter.bias;
            }
        }
    }

    output
}

pub fn conv_padded<const WEIGHTS: usize, const DEPTH: usize>(
    input: &Image,
    filters: &[Filter<WEIGHTS, DEPTH>],
) -> Image {
    assert_eq!(input.channels, DEPTH);
    let filter_size = filters[0].size;
    let padding = filter_size / 2;

    let mut output = Image {
        width: input.width,
        height: input.height,
        channels: filters.len(),
        pixels: vec![0.0; input.width * input.height * filters.len()],
    };

    for (filter_idx, filter) in filters.iter().enumerate() {
        for out_y in 0..input.height {
            for out_x in 0..input.width {
                let mut result = 0.0;

                for fy in 0..filter_size {
                    for fx in 0..filter_size {
                        for c in 0..DEPTH {
                            if let (Some(in_y), Some(in_x)) = (
                                (out_y + fy).checked_sub(padding),
                                (out_x + fx).checked_sub(padding),
                            ) {
                                if in_y < input.height && in_x < input.width {
                                    let input_idx =
                                        (in_y * input.width + in_x) * input.channels + c;
                                    let filter_idx_2d = fy * filter_size + fx;
                                    result +=
                                        input.pixels[input_idx] * filter.weights[c][filter_idx_2d];
                                }
                            }
                        }
                    }
                }

                let output_idx = (out_y * input.width + out_x) * filters.len() + filter_idx;
                output.pixels[output_idx] = result + filter.bias;
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
        for y in 3..64 {
            for x in 3..64 {
                let image = Image {
                    width: x,
                    height: y,
                    channels: 1,
                    pixels: (0..x * y).map(|p| p as f32).collect(),
                };
                let filter = vertical_sobel::<1>();
                conv(&image, &[filter]);
            }
        }
    }

    #[test]
    fn conv_10x3_2x2() {
        let image = Image {
            width: 10,
            height: 3,
            channels: 1,
            #[rustfmt::skip]
            pixels: vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2,
            ],
        };

        let filter = Filter::new([[1.0, -1.0, -1.0, 1.0]]);
        let result = conv(&image, &[filter]);

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
            channels: 1,
            #[rustfmt::skip]
            pixels: vec![
                0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 0.0, 0.1,
                0.2, 0.3, 0.4, 0.5,
            ],
        };

        let filter = Filter::new([[2.0]]);
        let result = conv(&image, &[filter]);

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
            channels: 1,
            #[rustfmt::skip]
            pixels: vec![
                0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0,
                0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1,
            ],
        };

        // Use a simple 3x3 box filter (all weights = 1/9)
        let filter = Filter::new([[1.0 / 9.0; 9]]);
        let result = conv(&image, &[filter]);

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
            channels: 1,
            #[rustfmt::skip]
            pixels: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0,
                0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0,
                0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0,
                0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0,
                0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        };
        let normal_image = Image {
            width: 5,
            height: 5,
            channels: 1,
            #[rustfmt::skip]
            pixels: vec![
                0.2, 0.3, 0.4, 0.5, 0.6,
                0.3, 0.4, 0.5, 0.6, 0.7,
                0.4, 0.5, 0.6, 0.7, 0.8,
                0.5, 0.6, 0.7, 0.8, 0.9,
                0.6, 0.7, 0.8, 0.9, 1.0,
            ],
        };

        let filter = Filter::new([[1.0 / 9.0; 9]]);

        let conv_result = conv(&padded_image, &[filter.clone()]);
        let conv_padded_result = super::conv_padded(&normal_image, &[filter]);

        assert_eq!(
            conv_result.pixels.as_slice(),
            conv_padded_result.pixels.as_slice()
        );
        assert_eq!(conv_result.width, conv_padded_result.width);
        assert_eq!(conv_result.height, conv_padded_result.height);
    }
}
