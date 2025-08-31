use super::Image;
use super::filter::{Filter, Filterable, GradientReceiver};
use super::layer::{BackPropagation, Layer};
use super::pixel::{Grayscale, Pixels};

#[derive(Debug)]
pub struct FeatureMap<'a, Data, Input, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize>
{
    data: Data,
    input: Option<Input>,
    filters: &'a mut [Filter<WEIGHTS, DEPTH, Grayscale, Grayscale>; WIDTH],
}

pub trait FeatureMapData<
    'a,
    Data,
    Input,
    const WEIGHTS: usize,
    const DEPTH: usize,
    const WIDTH: usize,
> where
    Self: Sized,
{
    fn feature_map(
        self,
        filters: &'a mut [Filter<WEIGHTS, DEPTH, Grayscale, Grayscale>; WIDTH],
    ) -> FeatureMap<'a, Self, Input, WEIGHTS, DEPTH, WIDTH>;
}

impl<'a, T, Input, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize>
    FeatureMapData<'a, T, Input, WEIGHTS, DEPTH, WIDTH> for T
{
    fn feature_map(
        self,
        filters: &'a mut [Filter<WEIGHTS, DEPTH, Grayscale, Grayscale>; WIDTH],
    ) -> FeatureMap<'a, T, Input, WEIGHTS, DEPTH, WIDTH> {
        FeatureMap {
            data: self,
            input: None,
            filters,
        }
    }
}

impl<'a, 'b, T, Input, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize> Layer<'b>
    for FeatureMap<'a, T, Input, WEIGHTS, DEPTH, WIDTH>
where
    T: Layer<'b, Item = Input>,
    Input: Filterable<WEIGHTS, DEPTH>,
{
    type Input = T::Input;
    type Item = FeatureSet<WIDTH>;

    fn forward(&mut self, input: Self::Input) -> Self::Item {
        let input = self.data.forward(input);
        let result = FeatureSet(self.filters.each_mut().map(|filter| input.filter(filter)));
        self.input = Some(input);
        result
    }
}

impl<'a, T, Input, const WEIGHTS: usize, const DEPTH: usize, const WIDTH: usize> BackPropagation
    for FeatureMap<'a, T, Input, WEIGHTS, DEPTH, WIDTH>
where
    T: BackPropagation<Gradient = <Input as GradientReceiver<WEIGHTS, DEPTH>>::Gradient>,
    Input: Filterable<WEIGHTS, DEPTH> + GradientReceiver<WEIGHTS, DEPTH>,
{
    type Gradient = FeatureSet<WIDTH>;

    fn backprop(&mut self, output_gradient: Self::Gradient) {
        let input = self
            .input
            .as_ref()
            .expect("`Layer::forward` must be called before `BackPropagation::backprop`");
        let learning_rate = self.learning_rate();
        let mut input_gradient = input.zero_gradient();

        for (filter, gradient) in self.filters.iter_mut().zip(output_gradient.iter()) {
            filter.backprop(learning_rate, input, gradient);
            input.accumulate_gradient(&mut input_gradient, gradient, filter);
        }

        self.data.backprop(input_gradient);
    }

    fn learning_rate(&self) -> f32 {
        self.data.learning_rate()
    }
}

#[derive(Debug, Clone)]
pub struct FeatureSet<const DEPTH: usize>([Image<Grayscale>; DEPTH]);

impl<const DEPTH: usize> Default for FeatureSet<DEPTH> {
    fn default() -> Self {
        Self(std::array::from_fn(|_| Image {
            width: 0,
            height: 0,
            pixels: Pixels::new(Vec::new()),
        }))
    }
}

impl<const DEPTH: usize> FeatureSet<DEPTH> {
    pub fn new(features: [Image<Grayscale>; DEPTH]) -> Self {
        Self(features)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Image<Grayscale>> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Image<Grayscale>> {
        self.0.iter_mut()
    }
}

impl<const DEPTH: usize> std::ops::Deref for FeatureSet<DEPTH> {
    type Target = [Image<Grayscale>; DEPTH];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const DEPTH: usize> std::ops::DerefMut for FeatureSet<DEPTH> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
