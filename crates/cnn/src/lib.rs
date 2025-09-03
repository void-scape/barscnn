#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod activation;
pub mod fc;
pub mod feature;
pub mod filter;
pub mod flatten;
pub mod image;
pub mod layer;
pub mod matrix;
pub mod pool;
pub mod rand;
pub mod softmax;

pub mod prelude {
    pub use super::activation::ReluLayer;
    pub use super::fc::{FcWeights, FullyConnectedLayer};
    pub use super::feature::FeatureMapLayer;
    pub use super::flatten::FlattenLayer;
    pub use super::image::Image;
    pub use super::matrix::Mat3d;
    pub use super::pool::MaxPoolLayer;
    pub use super::softmax::SoftmaxLayer;
    pub use super::{Cnn, Layer};
    pub use super::{filter, filter::Filter};
}

#[derive(Debug, Default, Clone)]
pub struct Cnn<const C: usize, const H: usize, const W: usize>(pub matrix::Mat3d<C, H, W>);

impl<const C: usize, const H: usize, const W: usize> Layer for Cnn<C, H, W> {
    type Input = matrix::Mat3d<C, H, W>;
    type Item = matrix::Mat3d<C, H, W>;

    fn input(&mut self, input: Self::Input) {
        self.0 = input;
    }

    fn forward(&mut self) -> Self::Item {
        self.0.clone()
    }

    fn backprop(&mut self, _: Self::Item, _: f32) {}
}

pub trait Layer {
    type Input;
    type Item;

    fn input(&mut self, input: Self::Input);

    fn forward(&mut self) -> Self::Item;

    fn backprop(&mut self, output_gradient: Self::Item, learning_rate: f32);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cnn_compile() {
        use prelude::*;

        let matrix = Mat3d::zero();
        let learning_rate = 0.001;
        let solution = 1;

        let mut cnn = Cnn::<1, 28, 28>::default()
            .feature_map_layer([
                filter::vertical_sobel(),
                filter::horizontal_sobel(),
                filter::gaussian_blur_3x3(),
            ])
            .leaky_relu_layer()
            .max_pool_layer::<2>()
            .flatten_layer()
            .fully_connected_layer::<2>(FcWeights::glorot(0))
            .softmax_layer();

        cnn.input(matrix);
        let mut result = cnn.forward();
        result[solution] = 1.0 - result[solution];
        cnn.backprop(result, learning_rate);
    }
}
