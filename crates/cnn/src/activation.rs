use std::marker::PhantomData;

use crate::Layer;
use crate::matrix::Mat3d;

pub const LEAKY_COEFF: f32 = 0.01;

#[derive(Debug, Clone)]
pub struct Activation<T, F, const C: usize, const H: usize, const W: usize> {
    pub layer: T,
    input: Mat3d<C, H, W>,
    _function: PhantomData<F>,
}

impl<T, F, const C: usize, const H: usize, const W: usize> Activation<T, F, C, H, W>
where
    F: ActivationFunction,
{
    pub fn activation(&self) -> Mat3d<C, H, W> {
        activation(&self.input, &mut F::map)
    }

    pub fn activation_backprop(&self, matrix: Mat3d<C, H, W>) -> Mat3d<C, H, W> {
        activation_backprop(&self.input, matrix, &mut F::backprop)
    }

    pub fn activation_mask(&self) -> Mat3d<C, H, W> {
        activation(&self.input, &mut F::pass_mask)
    }

    pub fn layer_input(&self) -> &Mat3d<C, H, W> {
        &self.input
    }
}

pub trait ReluLayer<const C: usize, const H: usize, const W: usize>
where
    Self: Sized,
{
    fn relu_layer(self) -> Activation<Self, Relu, C, H, W> {
        Activation {
            layer: self,
            input: Mat3d::zero(),
            _function: PhantomData,
        }
    }

    fn leaky_relu_layer(self) -> Activation<Self, LeakyRelu, C, H, W> {
        Activation {
            layer: self,
            input: Mat3d::zero(),
            _function: PhantomData,
        }
    }
}

impl<T, const C: usize, const H: usize, const W: usize> ReluLayer<C, H, W> for T where
    T: Layer<Item = Mat3d<C, H, W>>
{
}

pub trait ActivationFunction {
    fn map(input: &mut f32);

    fn backprop(input: f32, gradient: &mut f32);

    fn pass(input: f32) -> bool;

    fn pass_mask(input: &mut f32) {
        *input = Self::pass(*input) as u32 as f32;
    }
}

pub struct Relu;

impl ActivationFunction for Relu {
    fn map(input: &mut f32) {
        *input = input.max(0.0);
    }

    fn backprop(input: f32, gradient: &mut f32) {
        if input <= 0.0 {
            *gradient = 0.0;
        }
    }

    fn pass(input: f32) -> bool {
        input > 0.0
    }
}

pub struct LeakyRelu;

impl ActivationFunction for LeakyRelu {
    fn map(input: &mut f32) {
        *input = input.max(*input * LEAKY_COEFF);
    }

    fn backprop(input: f32, gradient: &mut f32) {
        if input <= 0.0 {
            *gradient *= LEAKY_COEFF;
        }
    }

    fn pass(_: f32) -> bool {
        true
    }
}

impl<T, F, const C: usize, const H: usize, const W: usize> Layer for Activation<T, F, C, H, W>
where
    T: Layer<Item = Mat3d<C, H, W>>,
    F: ActivationFunction,
{
    type Input = T::Input;
    type Item = Mat3d<C, H, W>;

    fn input(&mut self, input: Self::Input) {
        self.layer.input(input);
    }

    fn forward(&mut self) -> Self::Item {
        self.input = self.layer.forward();
        self.activation()
    }

    fn backprop(&mut self, output_gradient: Self::Item, learning_rate: f32) {
        self.layer
            .backprop(self.activation_backprop(output_gradient), learning_rate);
    }
}

pub fn activation<const C: usize, const H: usize, const W: usize, Map>(
    input: &Mat3d<C, H, W>,
    map: &mut Map,
) -> Mat3d<C, H, W>
where
    Map: FnMut(&mut f32),
{
    let mut output = input.clone();
    output.iter_mut().for_each(map);
    output
}

pub fn activation_backprop<const C: usize, const H: usize, const W: usize, Backprop>(
    original: &Mat3d<C, H, W>,
    input: Mat3d<C, H, W>,
    backprop: &mut Backprop,
) -> Mat3d<C, H, W>
where
    Backprop: FnMut(f32, &mut f32),
{
    let mut output = input;
    for (input, original) in output.iter_mut().zip(original.iter()) {
        backprop(*original, input);
    }
    output
}

#[cfg(test)]
mod test {
    use super::*;

    fn backprop<F: ActivationFunction>(_: F, mut gradient: f32, input: f32, expected: f32) {
        let g = &mut gradient;
        F::backprop(input, g);
        assert_eq!(*g, expected);
    }

    #[test]
    fn relu() {
        let input = Mat3d::<1, 1, 4>::new([-1.0, 0.0, 1.0, 2.0]);
        let output = input.clone().relu_layer().forward();
        assert_eq!(output, Mat3d::<1, 1, 4>::new([0.0, 0.0, 1.0, 2.0]));

        let output = input.leaky_relu_layer().forward();
        assert_eq!(
            output,
            Mat3d::<1, 1, 4>::new([-1.0 * LEAKY_COEFF, 0.0, 1.0, 2.0])
        );

        backprop(Relu, 10.0, 1.0, 10.0);
        backprop(Relu, 10.0, -1.0, 0.0);

        backprop(LeakyRelu, 10.0, 1.0, 10.0);
        backprop(LeakyRelu, 10.0, -1.0, 10.0 * LEAKY_COEFF);
    }
}
