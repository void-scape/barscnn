use super::Image;
use super::filter::{Filter, FilterData, Filterable};
use super::layer::{CachedLayer, Layer};
use super::pixel::Grayscale;

#[derive(Debug)]
pub struct FeatureMap<Feature, const DEPTH: usize>([Feature; DEPTH]);

pub trait FeatureMapData<'a, const WEIGHTS: usize, const DEPTH: usize>
where
    Self: Filterable,
{
    fn feature_map(
        &'a self,
        filters: &'a [Filter<WEIGHTS, Grayscale, Grayscale>; DEPTH],
    ) -> FeatureMap<FilterData<'a, &'a Self, WEIGHTS>, DEPTH>;
}

impl<'a, const WEIGHTS: usize, const DEPTH: usize> FeatureMapData<'a, WEIGHTS, DEPTH>
    for Image<Grayscale>
where
    Self: Filterable,
{
    fn feature_map(
        &'a self,
        filters: &'a [Filter<WEIGHTS, Grayscale, Grayscale>; DEPTH],
    ) -> FeatureMap<FilterData<'a, &'a Self, WEIGHTS>, DEPTH> {
        FeatureMap(
            filters
                .each_ref()
                .map(|filter| FilterData::new(self, filter)),
        )
    }
}

impl<T, const DEPTH: usize> Layer for FeatureMap<T, DEPTH>
where
    T: Layer<Item = Image<Grayscale>>,
    T::Cached: Layer<Item = Image<Grayscale>>,
{
    type Item = FeatureSet<DEPTH>;
    type Cached = FeatureMap<CachedLayer<T::Cached>, DEPTH>;

    fn forward(&self) -> Self::Item {
        FeatureSet(self.0.each_ref().map(|img| img.forward()))
    }

    fn forward_cached(self) -> CachedLayer<Self::Cached> {
        let data_cached = self.0.map(|d| d.forward_cached());
        let item = FeatureSet(data_cached.each_ref().map(|filter| filter.forward()));

        CachedLayer {
            layer: FeatureMap(data_cached),
            item,
        }
    }
}

#[derive(Debug, Clone)]
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
