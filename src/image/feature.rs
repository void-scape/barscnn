use super::Image;
use super::filter::{Filter, FilterData, Filterable};
use super::layer::{BackPropagation, Layer};
use super::pixel::{Grayscale, Pixels};

#[derive(Debug)]
pub struct FeatureMap<Feature, const DEPTH: usize>([Feature; DEPTH]);

pub trait FeatureMapData<'a, Input, const WEIGHTS: usize, const DEPTH: usize>
where
    Self: Filterable,
{
    fn feature_map(
        &'a self,
        filters: &'a mut [Filter<WEIGHTS, Grayscale, Grayscale>; DEPTH],
    ) -> FeatureMap<FilterData<'a, &'a Self, Input, WEIGHTS>, DEPTH>;
}

impl<'a, Input, const WEIGHTS: usize, const DEPTH: usize> FeatureMapData<'a, Input, WEIGHTS, DEPTH>
    for Image<Grayscale>
where
    Self: Filterable,
{
    fn feature_map(
        &'a self,
        filters: &'a mut [Filter<WEIGHTS, Grayscale, Grayscale>; DEPTH],
    ) -> FeatureMap<FilterData<'a, &'a Self, Input, WEIGHTS>, DEPTH> {
        FeatureMap(
            filters
                .each_mut()
                .map(|filter| FilterData::new(self, filter)),
        )
    }
}

impl<T, const DEPTH: usize> Layer for FeatureMap<T, DEPTH>
where
    T: Layer<Item = Image<Grayscale>>,
{
    type Item = FeatureSet<DEPTH>;

    fn forward(&mut self) -> Self::Item {
        FeatureSet(self.0.each_mut().map(|img| img.forward()))
    }
}

impl<T, const DEPTH: usize> BackPropagation for FeatureMap<T, DEPTH>
where
    T: Layer + BackPropagation<Gradient = Image<Grayscale>>,
{
    type Gradient = FeatureSet<DEPTH>;

    fn backprop(&mut self, output_gradient: Self::Gradient, learning_rate: f32) {
        for i in 0..DEPTH {
            // TODO: Move out of this array?
            self.0[i].backprop(output_gradient.0[i].clone(), learning_rate);
        }
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
}

impl<const DEPTH: usize> std::ops::Deref for FeatureSet<DEPTH> {
    type Target = [Image<Grayscale>; DEPTH];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
