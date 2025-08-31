use super::Image;
use super::feature::FeatureSet;
use super::layer::{BackPropagation, Layer};
use super::pixel::{Grayscale, PixelArray};

#[derive(Debug)]
pub struct Relu<Data, Input> {
    data: Data,
    input: Option<Input>,
}

pub trait ReluData<Input>
where
    Self: Sized,
{
    fn relu(self) -> Relu<Self, Input>;
}

impl<'a, T, Input> ReluData<Input> for T
where
    T: Layer<'a, Item = Input>,
{
    fn relu(self) -> Relu<Self, Input> {
        Relu {
            data: self,
            input: None,
        }
    }
}

impl<'a, T, Input> Layer<'a> for Relu<T, Input>
where
    T: Layer<'a, Item = Input>,
    Input: Activatable,
{
    type Input = T::Input;
    type Item = T::Item;

    fn forward(&mut self, input: Self::Input) -> Self::Item {
        let input = self.data.forward(input);
        let result = activation(&input, &mut |input| *input = input.max(0.0));
        self.input = Some(input);
        result
    }
}

impl<T, Input> BackPropagation for Relu<T, Input>
where
    T: BackPropagation<Gradient = Input>,
    Input: Activatable,
{
    type Gradient = Input;

    fn backprop(&mut self, mut output_gradient: Self::Gradient) {
        let input = self
            .input
            .as_ref()
            .expect("`Layer::forward` must be called before `BackPropagation::backprop`");

        pass_gradient(input, &mut output_gradient, &mut |input| input > 0.0);

        self.data.backprop(output_gradient);
    }

    fn learning_rate(&self) -> f32 {
        self.data.learning_rate()
    }
}

trait Activatable: Clone {
    fn iter(&self) -> impl Iterator<Item = &f32>;

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32>;

    fn zip(&mut self, other: &Self) -> impl Iterator<Item = (&mut f32, f32)> {
        self.iter_mut().zip(other.iter().copied())
    }
}

impl<const LEN: usize> Activatable for PixelArray<LEN> {
    fn iter(&self) -> impl Iterator<Item = &f32> {
        self.iter()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.iter_mut()
    }
}

impl Activatable for Image<Grayscale> {
    fn iter(&self) -> impl Iterator<Item = &f32> {
        self.pixels.iter()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.pixels.iter_mut()
    }
}

impl<const DEPTH: usize> Activatable for FeatureSet<DEPTH> {
    fn iter(&self) -> impl Iterator<Item = &f32> {
        self.iter().map(|img| img.pixels.iter()).flatten()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.iter_mut().map(|img| img.pixels.iter_mut()).flatten()
    }
}

fn activation<Input, Map>(input: &Input, map: &mut Map) -> Input
where
    Input: Activatable,
    Map: FnMut(&mut f32),
{
    let mut output = input.clone();
    output.iter_mut().for_each(map);
    output
}

fn pass_gradient<Input, Pass>(input: &Input, gradient: &mut Input, pass: &mut Pass)
where
    Input: Activatable,
    Pass: FnMut(f32) -> bool,
{
    gradient.zip(input).for_each(|(g, i)| {
        if !pass(i) {
            *g = 0.0;
        }
    });
}

#[cfg(test)]
mod test {
    use crate::image::pixel::PixelArray;

    use super::*;

    #[test]
    fn relu() {
        let pixels = [-1.0, 0.0, 1.0, 2.0];
        let input = PixelArray::new(pixels);

        let output = input.relu().forward(());
        assert_eq!(output.into_inner(), [0.0, 0.0, 1.0, 2.0]);

        let mut output = PixelArray::new([-1.0, 1.0, 2.0, -1.0]);
        pass_gradient(&input, &mut output, &mut |input| input > 0.0);

        assert_eq!(output.into_inner(), [0.0, 0.0, 2.0, -1.0]);
    }
}
