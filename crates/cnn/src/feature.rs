use crate::filter::{Filter, conv_padded};
use crate::image::Image;
use crate::layer::{BackPropagation, Layer};

#[derive(Debug, Clone)]
pub struct FeatureMap<Data, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize> {
    pub data: Data,
    pub filters: [Filter<WEIGHTS, DEPTH>; WIDTH],
    pub input: Image,
}

pub trait FeatureMapData<Data, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize>
where
    Self: Sized,
{
    fn feature_map(
        self,
        filters: [Filter<WEIGHTS, DEPTH>; WIDTH],
    ) -> FeatureMap<Self, WEIGHTS, DEPTH, WIDTH>;
}

impl<T, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize>
    FeatureMapData<T, WEIGHTS, DEPTH, WIDTH> for T
{
    fn feature_map(
        self,
        filters: [Filter<WEIGHTS, DEPTH>; WIDTH],
    ) -> FeatureMap<T, WEIGHTS, DEPTH, WIDTH> {
        FeatureMap {
            data: self,
            input: Image::default(),
            filters,
        }
    }
}

impl<T, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize> Layer
    for FeatureMap<T, WEIGHTS, DEPTH, WIDTH>
where
    T: Layer<Item = Image>,
{
    type Input = T::Input;
    type Item = Image;

    fn forward(&mut self, input: Self::Input) -> Self::Item {
        let input = self.data.forward(input);
        let result = conv_padded(&input, &self.filters);
        self.input = input;
        result
    }
}

impl<T, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize> BackPropagation
    for FeatureMap<T, WEIGHTS, DEPTH, WIDTH>
where
    T: BackPropagation<Gradient = Image>,
{
    type Gradient = Image;

    fn backprop(&mut self, output_gradient: Self::Gradient) {
        let input = &self.input;
        let mut gradient = Image {
            width: input.width,
            height: input.height,
            channels: input.channels,
            pixels: vec![0.0; input.pixels.len()],
        };

        let learning_rate = self.learning_rate();
        for (i, filter) in self.filters.iter_mut().enumerate() {
            filter.apply_gradients(input, &output_gradient, &mut gradient, i, learning_rate);
        }

        self.data.backprop(gradient);
    }

    fn learning_rate(&self) -> f32 {
        self.data.learning_rate()
    }
}
