use super::layer::{BackPropagation, Layer};
use super::pixel::PixelArray;

#[derive(Debug)]
pub struct FullyConnected<'a, Data, const INPUT: usize, const OUTPUT: usize> {
    data: Data,
    input: [f32; INPUT],
    fc: &'a mut FcWeights<INPUT, OUTPUT>,
}

pub trait FullyConnectedData<'a, const INPUT: usize, const OUTPUT: usize>
where
    Self: Sized,
{
    fn fully_connected(
        self,
        fc: &'a mut FcWeights<INPUT, OUTPUT>,
    ) -> FullyConnected<'a, Self, INPUT, OUTPUT>;
}

impl<'a, T, const INPUT: usize, const OUTPUT: usize> FullyConnectedData<'a, INPUT, OUTPUT> for T
where
    T: Layer<Item = PixelArray<INPUT>>,
{
    fn fully_connected(
        self,
        fc: &'a mut FcWeights<INPUT, OUTPUT>,
    ) -> FullyConnected<'a, Self, INPUT, OUTPUT> {
        FullyConnected {
            data: self,
            input: [0.0; INPUT],
            fc,
        }
    }
}

impl<'a, T, const INPUT: usize, const OUTPUT: usize> Layer for FullyConnected<'a, T, INPUT, OUTPUT>
where
    T: Layer<Item = PixelArray<INPUT>>,
{
    type Item = PixelArray<OUTPUT>;

    fn forward(&mut self) -> Self::Item {
        self.input = self.data.forward().into_inner();
        fully_connected(self.fc, self.input)
    }
}

impl<'a, Data, const INPUT: usize, const OUTPUT: usize> BackPropagation
    for FullyConnected<'a, Data, INPUT, OUTPUT>
where
    Data: BackPropagation<Gradient = PixelArray<INPUT>>,
{
    type Gradient = PixelArray<OUTPUT>;

    fn backprop(&mut self, output_gradient: Self::Gradient, learning_rate: f32) {
        for r in 0..OUTPUT {
            for c in 0..INPUT {
                let wgradient = self.input[c] * output_gradient[r];
                self.fc.weights[r][c] -= wgradient * learning_rate;
            }
        }

        for r in 0..OUTPUT {
            self.fc.bias[r] -= output_gradient[r] * learning_rate;
        }

        let mut out = [0.0; INPUT];
        for r in 0..OUTPUT {
            for c in 0..INPUT {
                out[c] += self.fc.weights[r][c] * output_gradient[r];
            }
        }

        self.data.backprop(PixelArray::new(out), learning_rate);
    }
}

fn fully_connected<const INPUT: usize, const OUTPUT: usize>(
    fc: &FcWeights<INPUT, OUTPUT>,
    pixels: [f32; INPUT],
) -> PixelArray<OUTPUT> {
    let mut output = [0.0; OUTPUT];
    for r in 0..OUTPUT {
        let mut result = 0.0;
        for c in 0..INPUT {
            result += fc.weights[r][c] * pixels[c];
        }
        result += fc.bias[r];
        output[r] = result;
    }
    PixelArray::new(output)
}

#[derive(Debug)]
pub struct FcWeights<const INPUT: usize, const OUTPUT: usize> {
    weights: [[f32; INPUT]; OUTPUT],
    bias: [f32; OUTPUT],
}

impl<const INPUT: usize, const OUTPUT: usize> FcWeights<INPUT, OUTPUT> {
    pub fn glorot() -> Self {
        glorot_initialization()
    }
}

fn glorot_initialization<const INPUT: usize, const OUTPUT: usize>() -> FcWeights<INPUT, OUTPUT> {
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

    let scale = (2.0 / (INPUT + OUTPUT) as f32).sqrt();
    let weights = [[0.0; INPUT].map(|_| {
        let random_val = rng.next() as f32 / u64::MAX as f32;
        (random_val - 0.5) * 2.0 * scale
    }); OUTPUT];

    let bias = [0.0; OUTPUT].map(|_| {
        let random_val = rng.next() as f32 / u64::MAX as f32;
        (random_val - 0.5) * 0.1
    });

    FcWeights { weights, bias }
}

#[derive(Debug)]
pub struct Softmax<Data> {
    data: Data,
}

impl<Data, const LEN: usize> Softmax<Data>
where
    Self: BackPropagation<Gradient = PixelArray<LEN>>,
{
    pub fn backprop_index(&mut self, index: usize, mut gradients: [f32; LEN], learning_rate: f32) {
        assert!(index < LEN);
        gradients[index] -= 1.0;
        self.backprop(PixelArray::new(gradients), learning_rate);
    }
}

pub trait SoftmaxData
where
    Self: Sized,
{
    fn softmax(self) -> Softmax<Self>;
}

impl<T, const OUTPUT: usize> SoftmaxData for T
where
    T: Layer<Item = PixelArray<OUTPUT>>,
{
    fn softmax(self) -> Softmax<Self> {
        Softmax { data: self }
    }
}

impl<T, const OUTPUT: usize> Layer for Softmax<T>
where
    T: Layer<Item = PixelArray<OUTPUT>>,
{
    type Item = [f32; OUTPUT];

    fn forward(&mut self) -> Self::Item {
        softmax(self.data.forward().into_inner())
    }
}

impl<T, const OUTPUT: usize> BackPropagation for Softmax<T>
where
    T: BackPropagation<Gradient = PixelArray<OUTPUT>>,
{
    type Gradient = PixelArray<OUTPUT>;

    fn backprop(&mut self, output_gradient: Self::Gradient, learning_rate: f32) {
        self.data.backprop(
            PixelArray::new(output_gradient.into_inner().map(|g| g * learning_rate)),
            learning_rate,
        );
    }
}

fn softmax<const OUTPUT: usize>(pixels: [f32; OUTPUT]) -> [f32; OUTPUT] {
    debug_assert!(OUTPUT != 0);
    debug_assert_eq!(pixels.len(), OUTPUT);

    let biggest = *pixels.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let exp = pixels.map(|v| (v - biggest).exp());
    let sum = exp.iter().sum::<f32>();
    if sum == 0.0 {
        [0.0; OUTPUT]
    } else {
        exp.map(|v| v / sum)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn reduce_3_2() {
        let weights = [[1.0, 0.5, -0.2], [-1.0, 2.0, 0.1]];
        let bias = [1.0, -0.5];

        let mut fc = FcWeights { weights, bias };
        let input = PixelArray::new([2.0, 3.0, -1.0]);
        let result = input.fully_connected(&mut fc).forward();

        assert_eq!(result.into_inner(), [4.7, 3.4]);
    }

    #[test]
    fn softmax() {
        let result = super::softmax([2.1, 0.8, 3.2, 1.5, 0.3, 2.8, 1.1, 0.6, 1.9, 2.4]);

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
