use super::layer::{CachedLayer, Layer};
use super::pixel::PixelArray;

#[derive(Debug)]
pub struct FullyConnected<'a, Data, const INPUT: usize, const OUTPUT: usize> {
    data: Data,
    fc: &'a FcWeights<INPUT, OUTPUT>,
}

pub trait FullyConnectedData<'a, const INPUT: usize, const OUTPUT: usize>
where
    Self: Sized,
{
    fn fully_connected(
        self,
        fc: &'a FcWeights<INPUT, OUTPUT>,
    ) -> FullyConnected<'a, Self, INPUT, OUTPUT>;
}

impl<'a, T, const INPUT: usize, const OUTPUT: usize> FullyConnectedData<'a, INPUT, OUTPUT> for T
where
    T: Layer<Item = PixelArray<INPUT>>,
{
    fn fully_connected(
        self,
        fc: &'a FcWeights<INPUT, OUTPUT>,
    ) -> FullyConnected<'a, Self, INPUT, OUTPUT> {
        FullyConnected { data: self, fc }
    }
}

impl<'a, T, const INPUT: usize, const OUTPUT: usize> Layer for FullyConnected<'a, T, INPUT, OUTPUT>
where
    T: Layer<Item = PixelArray<INPUT>>,
    T::Cached: Layer<Item = PixelArray<INPUT>>,
{
    type Item = PixelArray<OUTPUT>;
    type Cached = FullyConnected<'a, CachedLayer<T::Cached>, INPUT, OUTPUT>;

    fn forward(&self) -> Self::Item {
        fully_connected(self.fc, self.data.forward().into_inner())
    }

    fn forward_cached(self) -> CachedLayer<Self::Cached> {
        let data_cached = self.data.forward_cached();
        let item = fully_connected(self.fc, data_cached.item.into_inner());

        CachedLayer {
            layer: FullyConnected {
                fc: self.fc,
                data: data_cached,
            },
            item,
        }
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
    T::Cached: Layer<Item = PixelArray<OUTPUT>>,
{
    type Item = [f32; OUTPUT];
    type Cached = Softmax<CachedLayer<T::Cached>>;

    fn forward(&self) -> Self::Item {
        softmax(self.data.forward().into_inner())
    }

    fn forward_cached(self) -> CachedLayer<Self::Cached> {
        let data_cached = self.data.forward_cached();
        let item = softmax(data_cached.item.into_inner());

        CachedLayer {
            layer: Softmax { data: data_cached },
            item,
        }
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

        let fc = FcWeights { weights, bias };
        let input = PixelArray::new([2.0, 3.0, -1.0]);
        let result = input.fully_connected(&fc).forward();

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
