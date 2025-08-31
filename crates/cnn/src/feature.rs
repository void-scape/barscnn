use crate::filter::{Filter, conv_padded};
use crate::image::Image;
use crate::layer::{BackPropagation, Layer};

#[derive(Debug)]
pub struct FeatureMap<'a, Data, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize> {
    data: Data,
    input: Option<Image>,
    filters: &'a mut [Filter<WEIGHTS, DEPTH>; WIDTH],
}

pub trait FeatureMapData<'a, Data, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize>
where
    Self: Sized,
{
    fn feature_map(
        self,
        filters: &'a mut [Filter<WEIGHTS, DEPTH>; WIDTH],
    ) -> FeatureMap<'a, Self, WEIGHTS, DEPTH, WIDTH>;
}

impl<'a, T, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize>
    FeatureMapData<'a, T, WEIGHTS, DEPTH, WIDTH> for T
{
    fn feature_map(
        self,
        filters: &'a mut [Filter<WEIGHTS, DEPTH>; WIDTH],
    ) -> FeatureMap<'a, T, WEIGHTS, DEPTH, WIDTH> {
        FeatureMap {
            data: self,
            input: None,
            filters,
        }
    }
}

impl<'a, T, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize> Layer
    for FeatureMap<'a, T, WEIGHTS, DEPTH, WIDTH>
where
    T: Layer<Item = Image>,
{
    type Input = T::Input;
    type Item = Image;

    fn forward(&mut self, input: Self::Input) -> Self::Item {
        let input = self.data.forward(input);
        let result = conv_padded(&input, self.filters);
        self.input = Some(input);
        result
    }
}

impl<'a, T, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize> BackPropagation
    for FeatureMap<'a, T, WEIGHTS, DEPTH, WIDTH>
where
    T: BackPropagation<Gradient = Image>,
{
    type Gradient = Image;

    fn backprop(&mut self, output_gradient: Self::Gradient) {
        let input = self
            .input
            .as_ref()
            .expect("`Layer::forward` must be called before `BackPropagation::backprop`");
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
