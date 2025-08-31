use self::image::Image;
use self::layer::{BackPropagation, Layer};

pub mod activation;
pub mod feature;
pub mod filter;
pub mod flatten;
pub mod image;
pub mod layer;
pub mod linear;
pub mod pool;

pub mod prelude {
    pub use super::ImageCnn;
    pub use super::activation::ReluData;
    pub use super::feature::FeatureMapData;
    pub use super::flatten::FlattenData;
    pub use super::image::Image;
    pub use super::layer::Layer;
    pub use super::linear::{FcWeights, FullyConnectedData, SoftmaxData};
    pub use super::pool::MaxPoolData;
    pub use super::{filter, filter::Filter};
}

#[derive(Debug)]
pub struct ImageCnn {
    learning_rate: f32,
}

impl ImageCnn {
    pub fn learning_rate(lr: f32) -> Self {
        Self { learning_rate: lr }
    }
}

impl Layer for ImageCnn {
    type Input = Image;
    type Item = Image;

    fn forward(&mut self, input: Self::Input) -> Self::Item {
        input
    }
}

impl BackPropagation for ImageCnn {
    type Gradient = Image;

    fn backprop(&mut self, _: Self::Gradient) {}

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}
