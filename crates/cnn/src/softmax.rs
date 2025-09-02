use crate::Layer;
use crate::matrix::{Mat1d, Mat3d};

#[derive(Debug, Clone)]
pub struct Softmax<T, const OUT: usize> {
    pub layer: T,
    input: Mat1d<OUT>,
}

impl<T, const OUT: usize> Softmax<T, OUT> {
    pub fn softmax(&self) -> Mat1d<OUT> {
        softmax(&self.input)
    }
}

pub trait SoftmaxLayer<const OUT: usize>
where
    Self: Sized,
{
    fn softmax_layer(self) -> Softmax<Self, OUT>;
}

impl<T, const OUT: usize> SoftmaxLayer<OUT> for T
where
    T: Layer<Item = Mat1d<OUT>>,
{
    fn softmax_layer(self) -> Softmax<Self, OUT> {
        Softmax {
            layer: self,
            input: Mat3d::zero(),
        }
    }
}

impl<T, const OUT: usize> Layer for Softmax<T, OUT>
where
    T: Layer<Item = Mat1d<OUT>>,
{
    type Input = T::Input;
    type Item = Mat1d<OUT>;

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

pub fn softmax<const OUT: usize>(input: &Mat1d<OUT>) -> Mat1d<OUT> {
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
