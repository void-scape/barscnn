use crate::Layer;
use crate::matrix::{Mat1d, Mat3d};

#[derive(Debug, Clone)]
pub struct Softmax<T, const W: usize> {
    pub layer: T,
    input: Mat1d<W>,
}

impl<T, const W: usize> Softmax<T, W> {
    pub fn softmax(&self) -> Mat1d<W> {
        softmax(&self.input)
    }

    pub fn layer_input(&self) -> &Mat1d<W> {
        &self.input
    }
}

pub trait SoftmaxLayer<const W: usize>
where
    Self: Sized,
{
    fn softmax_layer(self) -> Softmax<Self, W>;
}

impl<T, const W: usize> SoftmaxLayer<W> for T
where
    T: Layer<Item = Mat1d<W>>,
{
    fn softmax_layer(self) -> Softmax<Self, W> {
        Softmax {
            layer: self,
            input: Mat3d::zero(),
        }
    }
}

impl<T, const W: usize> Layer for Softmax<T, W>
where
    T: Layer<Item = Mat1d<W>>,
{
    type Input = T::Input;
    type Item = Mat1d<W>;

    fn input(&mut self, input: Self::Input) {
        self.layer.input(input);
    }

    fn forward(&mut self) -> Self::Item {
        self.input = self.layer.forward();
        self.softmax()
    }

    fn backprop(&mut self, output_gradient: Self::Item, learning_rate: f32) {
        self.layer.backprop(output_gradient, learning_rate);
    }
}

pub fn softmax<const W: usize>(input: &Mat1d<W>) -> Mat1d<W> {
    let biggest = *input.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let mut exp = input.clone();
    exp.iter_mut().for_each(|v| *v = (*v - biggest).exp());
    let sum = exp.iter().sum::<f32>();
    if sum == 0.0 {
        Mat3d::zero()
    } else {
        exp.iter_mut().for_each(|v| *v /= sum);
        exp
    }
}

#[cfg(test)]
mod test {
    use crate::matrix::Mat3d;

    #[test]
    fn softmax() {
        let result = super::softmax::<10>(&Mat3d::new([
            2.1, 0.8, 3.2, 1.5, 0.3, 2.8, 1.1, 0.6, 1.9, 2.4,
        ]));

        let expected = [
            0.10241535,
            0.02791144,
            0.30767277,
            0.05620674,
            0.016929144,
            0.2062392,
            0.03767651,
            0.022851959,
            0.08385061,
            0.1382463,
        ];

        assert!((result.iter().sum::<f32>() - 1.0).abs() < f32::EPSILON);
        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
