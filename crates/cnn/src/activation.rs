use crate::image::Image;
use crate::layer::{BackPropagation, Layer};

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

impl<T, Input> ReluData<Input> for T
where
    T: Layer<Item = Input>,
{
    fn relu(self) -> Relu<Self, Input> {
        Relu {
            data: self,
            input: None,
        }
    }
}

impl<T, Input> Layer for Relu<T, Input>
where
    T: Layer<Item = Input>,
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

impl<const LEN: usize> Activatable for [f32; LEN] {
    fn iter(&self) -> impl Iterator<Item = &f32> {
        <_ as IntoIterator>::into_iter(self)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        <_ as IntoIterator>::into_iter(self)
    }
}

impl Activatable for Image {
    fn iter(&self) -> impl Iterator<Item = &f32> {
        self.pixels.iter()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.pixels.iter_mut()
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
    use super::*;

    #[test]
    fn relu() {
        let input = [-1.0, 0.0, 1.0, 2.0];

        let output = input.relu().forward(());
        assert_eq!(output, [0.0, 0.0, 1.0, 2.0]);

        let mut output = [-1.0, 1.0, 2.0, -1.0];
        pass_gradient(&input, &mut output, &mut |input| input > 0.0);

        assert_eq!(output, [0.0, 0.0, 2.0, -1.0]);
    }
}
