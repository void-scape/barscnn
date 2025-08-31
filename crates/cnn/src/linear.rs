use crate::layer::{BackPropagation, Layer};

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

impl<'a, T, const INPUT: usize, const OUTPUT: usize> FullyConnectedData<'a, INPUT, OUTPUT> for T {
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
    T: Layer<Item = [f32; INPUT]>,
{
    type Input = T::Input;
    type Item = [f32; OUTPUT];

    fn forward(&mut self, input: Self::Input) -> Self::Item {
        self.input = self.data.forward(input);
        fully_connected(self.fc, self.input)
    }
}

impl<'a, Data, const INPUT: usize, const OUTPUT: usize> BackPropagation
    for FullyConnected<'a, Data, INPUT, OUTPUT>
where
    Data: BackPropagation<Gradient = [f32; INPUT]>,
{
    type Gradient = [f32; OUTPUT];

    fn backprop(&mut self, output_gradient: Self::Gradient) {
        let learning_rate = self.learning_rate();

        for r in 0..OUTPUT {
            for c in 0..INPUT {
                self.fc.weights[r][c] -= learning_rate * self.input[c] * output_gradient[r];
            }
        }

        for r in 0..OUTPUT {
            self.fc.bias[r] -= learning_rate * output_gradient[r];
        }

        let mut out = [0.0; INPUT];
        for r in 0..OUTPUT {
            for c in 0..INPUT {
                out[c] += self.fc.weights[r][c] * output_gradient[r];
            }
        }

        self.data.backprop(out);
    }

    fn learning_rate(&self) -> f32 {
        self.data.learning_rate()
    }
}

fn fully_connected<const INPUT: usize, const OUTPUT: usize>(
    fc: &FcWeights<INPUT, OUTPUT>,
    pixels: [f32; INPUT],
) -> [f32; OUTPUT] {
    let mut output = [0.0; OUTPUT];
    for r in 0..OUTPUT {
        let mut result = 0.0;
        for c in 0..INPUT {
            result += fc.weights[r][c] * pixels[c];
        }
        result += fc.bias[r];
        output[r] = result;
    }
    output
}

#[derive(Debug)]
pub struct FcWeights<const INPUT: usize, const OUTPUT: usize> {
    weights: Box<[[f32; INPUT]; OUTPUT]>,
    bias: Box<[f32; OUTPUT]>,
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
    let weights_vec: Vec<[f32; INPUT]> = (0..OUTPUT)
        .map(|_| {
            [0.0; INPUT].map(|_| {
                let random_val = rng.next() as f32 / u64::MAX as f32;
                (random_val - 0.5) * 2.0 * scale
            })
        })
        .collect();

    let weights = weights_vec
        .into_boxed_slice()
        .try_into()
        .expect("Failed to convert to fixed-size array");

    let bias = vec![0.0; OUTPUT].into_boxed_slice().try_into().unwrap();

    FcWeights { weights, bias }
}

#[derive(Debug)]
pub struct Softmax<Data> {
    data: Data,
}

impl<Data, const LEN: usize> Softmax<Data>
where
    Self: BackPropagation<Gradient = [f32; LEN]>,
{
    pub fn backprop_index(&mut self, index: usize, mut gradients: [f32; LEN]) {
        assert!(index < LEN);
        gradients[index] -= 1.0;
        self.backprop(gradients);
    }
}

pub trait SoftmaxData<const OUTPUT: usize>
where
    Self: Sized,
{
    fn softmax(self) -> Softmax<Self>;
}

impl<T, const OUTPUT: usize> SoftmaxData<OUTPUT> for T
where
    T: Layer<Item = [f32; OUTPUT]>,
{
    fn softmax(self) -> Softmax<Self> {
        Softmax { data: self }
    }
}

impl<T, const OUTPUT: usize> Layer for Softmax<T>
where
    T: Layer<Item = [f32; OUTPUT]>,
{
    type Input = T::Input;
    type Item = [f32; OUTPUT];

    fn forward(&mut self, input: Self::Input) -> Self::Item {
        softmax(self.data.forward(input))
    }
}

impl<T, const OUTPUT: usize> BackPropagation for Softmax<T>
where
    T: BackPropagation<Gradient = [f32; OUTPUT]>,
{
    type Gradient = [f32; OUTPUT];

    fn backprop(&mut self, output_gradient: Self::Gradient) {
        self.data.backprop(output_gradient);
    }

    fn learning_rate(&self) -> f32 {
        self.data.learning_rate()
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

        let mut fc = FcWeights {
            weights: Box::new(weights),
            bias: Box::new(bias),
        };
        let input = [2.0, 3.0, -1.0];
        let result = input.fully_connected(&mut fc).forward(());
        assert_eq!(result, [4.7, 3.4]);
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
