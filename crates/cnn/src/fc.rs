use crate::Layer;
use crate::matrix::{Mat1d, Mat2d, Mat3d};

#[derive(Debug, Clone)]
pub struct FullyConnected<T, const H: usize, const W: usize> {
    pub layer: T,
    fc: FcWeights<H, W>,
    input: Mat1d<H>,
}

impl<T, const H: usize, const W: usize> FullyConnected<T, H, W> {
    #[doc(hidden)]
    const SIZE_CONSTRAINTS: () = assert!(H >= W, "output count is greater than input count");

    pub fn fully_connected(&self) -> Mat1d<W> {
        fully_connected(&self.fc, &self.input)
    }
}

pub trait FullyConnectedLayer<const H: usize>
where
    Self: Sized,
{
    fn fully_connected_layer<const W: usize>(
        self,
        fc: FcWeights<H, W>,
    ) -> FullyConnected<Self, H, W>;
}

impl<T, const H: usize> FullyConnectedLayer<H> for T {
    fn fully_connected_layer<const W: usize>(
        self,
        fc: FcWeights<H, W>,
    ) -> FullyConnected<Self, H, W> {
        let _invalid_output_size = FullyConnected::<Self, H, W>::SIZE_CONSTRAINTS;
        FullyConnected {
            layer: self,
            fc,
            input: Mat3d::zero(),
        }
    }
}

impl<T, const H: usize, const W: usize> Layer for FullyConnected<T, H, W>
where
    T: Layer<Item = Mat1d<H>>,
{
    type Input = T::Input;
    type Item = Mat1d<W>;

    fn input(&mut self, input: Self::Input) {
        self.layer.input(input);
    }

    fn forward(&mut self) -> Self::Item {
        self.input = self.layer.forward();
        self.fully_connected()
    }

    fn backprop(&mut self, output_gradient: Self::Item, learning_rate: f32) {
        // Source: https://victorzhou.com/blog/intro-to-cnns-part-2/

        // Compute gradient with respect to filter weights.
        for h in 0..H {
            for w in 0..W {
                let g = self.input[h] * output_gradient[w];
                *self.fc.weights.chw_mut(0, h, w) -= learning_rate * g;
            }
        }

        for w in 0..W {
            self.fc.bias[w] -= learning_rate * output_gradient[w];
        }

        // Propagate gradients backward to the input.
        //
        // This computes how much each input value should change to reduce the loss.
        let mut input_gradient = Mat1d::zero();
        for h in 0..H {
            let mut grad = 0.0;
            for w in 0..W {
                grad += self.fc.weights.chw(0, h, w) * output_gradient[w];
            }
            input_gradient[h] = grad;
        }
        self.layer.backprop(input_gradient, learning_rate);
    }
}

pub fn fully_connected<const H: usize, const W: usize>(
    fc: &FcWeights<H, W>,
    input: &Mat1d<H>,
) -> Mat1d<W> {
    let mut output = Mat1d::zero();
    for w in 0..W {
        let mut result = 0.0;
        for h in 0..H {
            result += fc.weights.chw(0, h, w) * input[h];
        }
        result += fc.bias[w];
        output[w] = result;
    }
    output
}

#[derive(Debug, Clone)]
pub struct FcWeights<const H: usize, const W: usize> {
    weights: Mat2d<H, W>,
    bias: Mat1d<W>,
}

impl<const H: usize, const W: usize> FcWeights<H, W> {
    pub fn glorot() -> Self {
        glorot_initialization()
    }
}

fn glorot_initialization<const H: usize, const W: usize>() -> FcWeights<H, W> {
    struct XorShiftRng {
        state: u64,
    }

    impl XorShiftRng {
        fn new(seed: u64) -> Self {
            Self {
                state: if seed == 0 { 1 } else { seed },
            }
        }

        fn next(&mut self) -> u64 {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            self.state
        }
    }

    fn time_seed() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    let mut rng = XorShiftRng::new(time_seed());

    let scale = (2.0 / (H + W) as f32).sqrt();
    let weights = Mat3d::new((0..H * W).map(|_| {
        let random_val = rng.next() as f32 / u64::MAX as f32;
        (random_val - 0.5) * 2.0 * scale
    }));
    let bias = Mat3d::zero();

    FcWeights { weights, bias }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn fc_3_2() {
        let weights = [1.0, 0.5, -0.2, -1.0, 2.0, 0.1];
        let bias = [1.0, -0.5];

        let fc = FcWeights {
            weights: Mat3d::<1, 3, 2>::new(weights),
            bias: Mat3d::<1, 1, 2>::new(bias),
        };
        let input = Mat3d::<1, 1, 3>::new([2.0, 3.0, -1.0]);
        let result = input.fully_connected_layer(fc).forward();
        assert_eq!(result.as_slice(), &[0.39999998, -2.6]);
    }
}
