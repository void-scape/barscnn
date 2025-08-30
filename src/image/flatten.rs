use std::marker::PhantomData;

use super::Image;
use super::feature::FeatureSet;
use super::layer::{CachedLayer, Layer};
use super::pixel::{Grayscale, PixelArray};

#[derive(Debug)]
pub struct Flatten<Data, const LEN: usize> {
    data: Data,
    _len: PhantomData<[Data; LEN]>,
}

pub trait FlattenData<const LEN: usize>
where
    Self: Sized,
{
    fn flatten(self) -> Flatten<Self, LEN>;
}

impl<T, const LEN: usize> FlattenData<LEN> for T
where
    T: Layer,
    T::Item: Flattenable<LEN>,
{
    fn flatten(self) -> Flatten<Self, LEN> {
        Flatten {
            data: self,
            _len: PhantomData,
        }
    }
}

impl<T, const LEN: usize> Layer for Flatten<T, LEN>
where
    T: Layer,
    T::Item: Flattenable<LEN>,
    <T::Cached as Layer>::Item: Flattenable<LEN>,
{
    type Item = PixelArray<LEN>;
    type Cached = Flatten<CachedLayer<T::Cached>, LEN>;

    fn forward(&self) -> Self::Item {
        self.data.forward().flatten()
    }

    fn forward_cached(self) -> CachedLayer<Self::Cached> {
        let data_cached = self.data.forward_cached();
        let item = data_cached.item.clone().flatten();

        CachedLayer {
            layer: Flatten {
                data: data_cached,
                _len: PhantomData,
            },
            item,
        }
    }
}

trait Flattenable<const LEN: usize>: Clone {
    fn flatten(&self) -> PixelArray<LEN>;
}

impl<const LEN: usize> Flattenable<LEN> for Image<Grayscale> {
    fn flatten(&self) -> PixelArray<LEN> {
        flatten(self)
    }
}

impl<const LEN: usize, const DEPTH: usize> Flattenable<LEN> for FeatureSet<DEPTH> {
    fn flatten(&self) -> PixelArray<LEN> {
        flatten_features(self)
    }
}

fn flatten<const OUTPUT: usize>(image: &Image<Grayscale>) -> PixelArray<OUTPUT> {
    if image.pixels.len() != OUTPUT {
        panic!(
            "called `flatten` with the incorrect `OUTPUT`, expected {}, got {OUTPUT}",
            image.pixels.len()
        );
    }

    let mut arr = [0.0; OUTPUT];
    arr.copy_from_slice(image.pixels.as_slice());
    PixelArray::new(arr)
}

fn flatten_features<const DEPTH: usize, const OUTPUT: usize>(
    features: &FeatureSet<DEPTH>,
) -> PixelArray<OUTPUT> {
    debug_assert_eq!(
        features.iter().map(|img| img.pixels.len()).sum::<usize>(),
        OUTPUT
    );

    let mut arr = [0.0; OUTPUT];
    let mut offset = 0;
    for img in features.iter() {
        arr[offset..offset + img.pixels.len()].copy_from_slice(img.pixels.as_slice());
        offset += img.pixels.len();
    }
    PixelArray::new(arr)
}
