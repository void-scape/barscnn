pub trait Layer: Sized {
    type Item;
    type Cached: Layer;

    fn forward(&self) -> Self::Item;

    fn forward_cached(self) -> CachedLayer<Self::Cached>;
}

#[derive(Debug)]
pub struct CachedLayer<T: Layer> {
    pub layer: T,
    pub item: T::Item,
}

impl<T> Layer for CachedLayer<T>
where
    T: Layer,
    T::Item: Clone,
{
    type Item = T::Item;
    type Cached = T;

    fn forward(&self) -> Self::Item {
        self.item.clone()
    }

    fn forward_cached(self) -> CachedLayer<Self::Cached> {
        self
    }
}
