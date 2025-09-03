use crate::matrix::Mat3d;
use crate::rand::XorShiftRng;

#[derive(Debug, Clone)]
pub struct Filter<const SIZE: usize, const C: usize> {
    pub weights: Mat3d<C, SIZE, SIZE>,
    pub bias: f32,
}

impl<const SIZE: usize, const C: usize> Filter<SIZE, C> {
    pub fn new(weights: Mat3d<C, SIZE, SIZE>) -> Self {
        Self { weights, bias: 0.0 }
    }

    pub fn from_2d_array(weights: [f32; SIZE * SIZE]) -> Self {
        Self {
            weights: Mat3d::new((0..C).flat_map(|_| weights.iter().copied())),
            bias: 0.0,
        }
    }
}

pub fn feature_set<const LEN: usize, const SIZE: usize, const C: usize>(
    f: impl FnMut(usize) -> Filter<SIZE, C>,
) -> [Filter<SIZE, C>; LEN] {
    std::array::from_fn(f)
}

pub fn xavier<const SIZE: usize, const C: usize>(seed: u64) -> Filter<SIZE, C> {
    let mut rng = XorShiftRng::new(seed);
    let fan_in = SIZE * SIZE * C;
    let fan_out = SIZE * SIZE;
    let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();
    let weights = Mat3d::new((0..C * SIZE * SIZE).map(|_| rng.uniform() * scale));
    Filter { weights, bias: 0.0 }
}

pub fn identity<const C: usize>() -> Filter<3, C> {
    #[rustfmt::skip]
    let weights = [
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ];

    Filter::from_2d_array(weights)
}

pub fn uniform_3x3<const C: usize>() -> Filter<3, C> {
    Filter::from_2d_array([1.0 / 6.0; 9])
}

pub fn vertical_sobel<const C: usize>() -> Filter<3, C> {
    #[rustfmt::skip]
    let weights = [
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ];

    Filter::from_2d_array(weights)
}

pub fn horizontal_sobel<const C: usize>() -> Filter<3, C> {
    #[rustfmt::skip]
    let weights = [
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ];

    Filter::from_2d_array(weights)
}

pub fn gaussian_blur_3x3<const C: usize>() -> Filter<3, C> {
    #[rustfmt::skip]
    let weights = [
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0,
    ];

    Filter::from_2d_array(weights.map(|v| v / 16.0))
}

pub fn laplacian_3x3<const C: usize>() -> Filter<3, C> {
    #[rustfmt::skip]
    let weights = [
         0.0, -1.0,  0.0,
        -1.0,  4.0, -1.0,
         0.0, -1.0,  0.0,
    ];

    Filter::from_2d_array(weights)
}

pub fn apply_filter_gradients<
    const SIZE: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const FILTERS: usize,
>(
    filters: &mut [Filter<SIZE, C>; FILTERS],
    input: &Mat3d<C, H, W>,
    output_gradients: &Mat3d<FILTERS, { H - SIZE + 1 }, { W - SIZE + 1 }>,
    learning_rate: f32,
) -> Mat3d<C, H, W> {
    let mut input_gradients = Mat3d::zero();

    for (filter_idx, filter) in filters.iter_mut().enumerate() {
        let mut filter_gradients = Mat3d::<C, SIZE, SIZE>::zero();

        for oy in 0..H - SIZE + 1 {
            for ox in 0..W - SIZE + 1 {
                let g = output_gradients.chw(filter_idx, oy, ox);

                for fy in 0..SIZE {
                    for fx in 0..SIZE {
                        for c in 0..C {
                            // Source: https://victorzhou.com/blog/intro-to-cnns-part-2/

                            // Compute gradient with respect to filter weights using the chain rule:
                            // ∂L/∂W = ∂L/∂Out * ∂Out/∂W = output_gradient * input
                            *filter_gradients.chw_mut(c, fy, fx) +=
                                input.chw(c, oy + fy, ox + fx) * g;

                            // Propagate gradients backward to the input using the chain rule:
                            // ∂L/∂In = ∂L/∂Out * ∂Out/∂In = output_gradient * filter_weight
                            //
                            // This computes how much each input value should change to reduce the loss.
                            *input_gradients.chw_mut(c, oy + fy, ox + fx) +=
                                filter.weights.chw(c, fy, fx) * g;
                        }
                    }
                }
            }
        }

        // `filter_gradients` tells us how much each filter weight should change to reduce the loss.
        filter
            .weights
            .iter_mut()
            .zip(filter_gradients.iter())
            .for_each(|(weight, grad)| *weight -= learning_rate * *grad);
    }

    input_gradients
}

pub fn conv<
    const SIZE: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const FILTERS: usize,
>(
    input: &Mat3d<C, H, W>,
    filters: &[Filter<SIZE, C>; FILTERS],
) -> Mat3d<FILTERS, { H - SIZE + 1 }, { W - SIZE + 1 }> {
    let out_height = H - SIZE + 1;
    let out_width = W - SIZE + 1;

    let mut output = Mat3d::zero();
    for (filter_idx, filter) in filters.iter().enumerate() {
        for oy in 0..out_height {
            for ox in 0..out_width {
                let mut result = 0.0;
                for fy in 0..SIZE {
                    for fx in 0..SIZE {
                        for c in 0..C {
                            result +=
                                input.chw(c, oy + fy, ox + fx) * filter.weights.chw(c, fy, fx);
                        }
                    }
                }
                *output.chw_mut(filter_idx, oy, ox) = result + filter.bias;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Mat3d;

    #[test]
    fn conv_identity_filter() {
        #[rustfmt::skip]
        let input = Mat3d::<1, 3, 3>::new([
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);

        let output = conv(&input, &[identity::<1>()]);
        assert_eq!(output.chw(0, 0, 0), 5.0);
    }

    #[test]
    fn conv_uniform_3x3_filter() {
        #[rustfmt::skip]
        let input = Mat3d::<1, 4, 4>::new([
            1.0,  2.0,  3.0,  4.0,
            5.0,  6.0,  7.0,  8.0,
            9.0, 10.0, 11.0, 12.0,
           13.0, 14.0, 15.0, 16.0,
        ]);
        let output = conv(&input, &[uniform_3x3::<1>()]);

        // Top-left 3x3 region: [1,2,3,5,6,7,9,10,11]
        // Sum = 54, so output[0,0] = 54 * (1/6) = 9.0
        let expected_top_left =
            (1.0 + 2.0 + 3.0 + 5.0 + 6.0 + 7.0 + 9.0 + 10.0 + 11.0) * (1.0 / 6.0);
        assert_eq!(output.chw(0, 0, 0), expected_top_left);

        // Top-right 3x3 region: [2,3,4,6,7,8,10,11,12]
        // Sum = 63, so output[0,1] = 63 * (1/6) = 10.5
        let expected_top_right =
            (2.0 + 3.0 + 4.0 + 6.0 + 7.0 + 8.0 + 10.0 + 11.0 + 12.0) * (1.0 / 6.0);
        assert_eq!(output.chw(0, 0, 1), expected_top_right);

        // Bottom-left 3x3 region: [5,6,7,9,10,11,13,14,15]
        // Sum = 90, so output[1,0] = 90 * (1/6) = 15.0
        let expected_bottom_left =
            (5.0 + 6.0 + 7.0 + 9.0 + 10.0 + 11.0 + 13.0 + 14.0 + 15.0) * (1.0 / 6.0);
        assert_eq!(output.chw(0, 1, 0), expected_bottom_left);

        // Bottom-right 3x3 region: [6,7,8,10,11,12,14,15,16]
        // Sum = 99, so output[1,1] = 99 * (1/6) = 16.5
        let expected_bottom_right =
            (6.0 + 7.0 + 8.0 + 10.0 + 11.0 + 12.0 + 14.0 + 15.0 + 16.0) * (1.0 / 6.0);
        assert_eq!(output.chw(0, 1, 1), expected_bottom_right);
    }

    #[test]
    fn conv_vertical_sobel_edge_detection() {
        #[rustfmt::skip]
        let input = Mat3d::<1, 4, 4>::new([
            1.0, 1.0, 5.0, 5.0,
            1.0, 1.0, 5.0, 5.0,
            1.0, 1.0, 5.0, 5.0,
            1.0, 1.0, 5.0, 5.0,
        ]);

        let filters = [vertical_sobel::<1>()];
        let output = conv(&input, &filters);

        // Vertical Sobel filter weights:
        // [-1, 0, 1]
        // [-2, 0, 2]
        // [-1, 0, 1]

        // For top-left 3x3 region:
        // [1, 1, 5]
        // [1, 1, 5]
        // [1, 1, 5]
        // Result = (-1*1 + 0*1 + 1*5) + (-2*1 + 0*1 + 2*5) + (-1*1 + 0*1 + 1*5)
        //        = (-1 + 0 + 5) + (-2 + 0 + 10) + (-1 + 0 + 5)
        //        = 4 + 8 + 4 = 16
        assert_eq!(output.chw(0, 0, 0), 16.0);

        // For top-right 3x3 region:
        // [1, 5, 5]
        // [1, 5, 5]
        // [1, 5, 5]
        // Result = (-1*1 + 0*5 + 1*5) + (-2*1 + 0*5 + 2*5) + (-1*1 + 0*5 + 1*5)
        //        = (-1 + 0 + 5) + (-2 + 0 + 10) + (-1 + 0 + 5)
        //        = 4 + 8 + 4 = 16
        assert_eq!(output.chw(0, 0, 1), 16.0);

        // For bottom-left 3x3 region (same pattern):
        assert_eq!(output.chw(0, 1, 0), 16.0);

        // For bottom-right 3x3 region (same pattern):
        assert_eq!(output.chw(0, 1, 1), 16.0);
    }
}
