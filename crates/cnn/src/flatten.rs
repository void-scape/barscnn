use crate::Layer;
use crate::matrix::Mat3d;

#[derive(Debug, Clone)]
pub struct Flatten<T, const C: usize, const H: usize, const W: usize> {
    pub layer: T,
    input: Mat3d<C, H, W>,
}

impl<T, const C: usize, const H: usize, const W: usize> Flatten<T, C, H, W> {
    pub fn flatten(&self) -> Mat3d<1, 1, { C * H * W }> {
        self.input.clone().reshape()
    }

    pub fn unflatten(&self, matrix: Mat3d<1, 1, { C * H * W }>) -> Mat3d<C, H, W> {
        matrix.reshape()
    }
}

pub trait FlattenLayer<const C: usize, const H: usize, const W: usize>
where
    Self: Sized,
{
    fn flatten_layer(self) -> Flatten<Self, C, H, W>;
}

impl<T, const C: usize, const H: usize, const W: usize> FlattenLayer<C, H, W> for T
where
    T: Layer,
{
    fn flatten_layer(self) -> Flatten<Self, C, H, W> {
        Flatten {
            layer: self,
            input: Mat3d::zero(),
        }
    }
}

impl<T, const C: usize, const H: usize, const W: usize> Layer for Flatten<T, C, H, W>
where
    T: Layer<Item = Mat3d<C, H, W>>,
    [(); C * H * W]:,
{
    type Input = T::Input;
    type Item = Mat3d<1, 1, { C * H * W }>;

    fn input(&mut self, input: Self::Input) {
        self.layer.input(input);
    }

    fn forward(&mut self) -> Self::Item {
        let input = self.layer.forward();
        self.input = input.clone();
        self.flatten()
    }

    fn backprop(&mut self, output_gradient: Self::Item, learning_rate: f32) {
        self.layer
            .backprop(self.unflatten(output_gradient), learning_rate);
    }
}
