pub trait Layer {
    type Item;

    fn forward(&mut self) -> Self::Item;
}
