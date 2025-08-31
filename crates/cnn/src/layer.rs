pub trait Layer: Sized {
    type Input;
    type Item;

    fn forward(&mut self, input: Self::Input) -> Self::Item;
}

pub trait BackPropagation {
    type Gradient;

    fn backprop(&mut self, output_gradient: Self::Gradient);

    fn learning_rate(&self) -> f32;
}

#[cfg(test)]
mod test {
    use super::*;

    impl<const LEN: usize> Layer for [f32; LEN] {
        type Input = ();
        type Item = Self;

        fn forward(&mut self, _: Self::Input) -> Self::Item {
            *self
        }
    }

    impl<const LEN: usize> BackPropagation for [f32; LEN] {
        type Gradient = Self;

        fn backprop(&mut self, _: Self::Gradient) {
            #[cfg(not(debug_assertions))]
            panic!("This can not be called in production");
        }

        fn learning_rate(&self) -> f32 {
            #[cfg(not(debug_assertions))]
            panic!("This can not be called in production");
            #[cfg(debug_assertions)]
            0.0
        }
    }
}
