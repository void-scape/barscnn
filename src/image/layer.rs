pub trait Layer: Sized {
    type Item;

    fn forward(&mut self) -> Self::Item;
}

pub trait BackPropagation {
    type Gradient;

    fn backprop(&mut self, output_gradient: Self::Gradient, learning_rate: f32);
}
