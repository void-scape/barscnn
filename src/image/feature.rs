use super::Image;
use super::filter::{Filter, FilterImage, FilteredImage};
use super::layer::Layer;
use super::pixel::{Grayscale, Pixel};

#[derive(Debug)]
pub struct FeatureMap<'a, Pixel, const WEIGHTS: usize, const DEPTH: usize>(
    [FilteredImage<'a, Pixel, WEIGHTS>; DEPTH],
);

pub trait FeatureMapImage<'a, T, const WEIGHTS: usize, const DEPTH: usize>
where
    T: Pixel,
{
    fn feature_map(
        &'a self,
        filters: &'a [Filter<WEIGHTS, T, Grayscale>; DEPTH],
    ) -> FeatureMap<'a, T, WEIGHTS, DEPTH>;
}

impl<'a, T, const WEIGHTS: usize, const DEPTH: usize> FeatureMapImage<'a, T, WEIGHTS, DEPTH>
    for Image<T>
where
    T: Pixel,
{
    fn feature_map(
        &'a self,
        filters: &'a [Filter<WEIGHTS, T, Grayscale>; DEPTH],
    ) -> FeatureMap<'a, T, WEIGHTS, DEPTH> {
        FeatureMap(filters.each_ref().map(|filter| self.filter(filter)))
    }
}

impl<'a, T, const WEIGHTS: usize, const DEPTH: usize> Layer for FeatureMap<'a, T, WEIGHTS, DEPTH>
where
    T: Pixel,
{
    type Item = FeatureSet<DEPTH>;

    fn forward(&mut self) -> Self::Item {
        FeatureSet(self.0.each_mut().map(|img| img.forward()))
    }
}

#[derive(Debug)]
pub struct FeatureSet<const DEPTH: usize>([Image<Grayscale>; DEPTH]);

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
