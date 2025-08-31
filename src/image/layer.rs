pub trait Layer<'a>: Sized {
    type Input: Clone + 'a;
    type Item;

    fn forward(&mut self, input: Self::Input) -> Self::Item;
}

pub trait BackPropagation {
    type Gradient;

    fn backprop(&mut self, output_gradient: Self::Gradient);

    fn learning_rate(&self) -> f32;
}
