use super::filter::Filter;
use super::pixel::{Grayscale, Pixel, PixelArray};
use super::{Image, linear, pool};

#[derive(Debug)]
pub struct FeatureMap<const DEPTH: usize>([Image<Grayscale>; DEPTH]);

impl<const DEPTH: usize> FeatureMap<DEPTH> {
    pub fn from_filters<const WEIGHTS: usize, T>(
        image: &Image<T>,
        filters: &[Filter<WEIGHTS, T, Grayscale>; DEPTH],
    ) -> Self
    where
        T: Pixel,
    {
        Self(filters.each_ref().map(|f| f.forward(image)))
    }

    pub fn max_pool(&self, size: usize) -> Self {
        Self(self.0.each_ref().map(|img| pool::max_pool(size, img)))
    }

    pub fn flatten<const LEN: usize>(&self) -> PixelArray<LEN> {
        linear::flatten_features(self)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Image<Grayscale>> {
        self.0.iter()
    }
}
