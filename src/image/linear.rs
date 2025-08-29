use super::Image;
use super::feature::FeatureMap;
use super::pixel::{Grayscale, PixelArray};

pub fn flatten<const OUTPUT: usize>(image: &Image<Grayscale>) -> PixelArray<OUTPUT> {
    if image.pixels.len() != OUTPUT {
        panic!(
            "called `flatten` with the incorrect `OUTPUT`, expected {}, got {OUTPUT}",
            image.pixels.len()
        );
    }

    let mut arr = [0.0; OUTPUT];
    arr.copy_from_slice(image.pixels.as_slice());
    PixelArray::new(arr)
}

pub fn flatten_features<const DEPTH: usize, const OUTPUT: usize>(
    features: &FeatureMap<DEPTH>,
) -> PixelArray<OUTPUT> {
    debug_assert_eq!(
        features.iter().map(|img| img.pixels.len()).sum::<usize>(),
        OUTPUT
    );

    let mut arr = [0.0; OUTPUT];
    let mut offset = 0;
    for img in features.iter() {
        arr[offset..offset + img.pixels.len()].copy_from_slice(img.pixels.as_slice());
        offset += img.pixels.len();
    }
    PixelArray::new(arr)
}

#[derive(Debug)]
pub struct FullyConnected<const INPUT: usize, const OUTPUT: usize> {
    weights: [[f32; INPUT]; OUTPUT],
    bias: [f32; OUTPUT],
}

impl<const INPUT: usize, const OUTPUT: usize> FullyConnected<INPUT, OUTPUT> {
    pub fn new(weights: [[f32; INPUT]; OUTPUT], bias: [f32; OUTPUT]) -> Self {
        Self { weights, bias }
    }

    pub fn glorot() -> Self {
        glorot_initialization()
    }

    pub fn forward(&self, pixels: &PixelArray<INPUT>) -> PixelArray<OUTPUT> {
        fully_connected(self, pixels.as_slice())
    }
}

fn glorot_initialization<const INPUT: usize, const OUTPUT: usize>() -> FullyConnected<INPUT, OUTPUT>
{
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

    FullyConnected::new(weights, bias)
}

fn fully_connected<const INPUT: usize, const OUTPUT: usize>(
    fc: &FullyConnected<INPUT, OUTPUT>,
    pixels: &[f32],
) -> PixelArray<OUTPUT> {
    debug_assert_eq!(pixels.len(), INPUT);

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

pub fn softmax<const OUTPUT: usize>(pixels: &PixelArray<OUTPUT>) -> PixelArray<OUTPUT> {
    debug_assert!(OUTPUT != 0);
    debug_assert_eq!(pixels.len(), OUTPUT);

    let pixels = pixels.into_inner();
    let biggest = *pixels.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let exp = pixels.map(|v| (biggest - v).exp());
    let sum = exp.iter().sum::<f32>();
    if sum == 0.0 {
        PixelArray::default()
    } else {
        PixelArray::new(exp.map(|v| v / sum))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn reduce_3_2() {
        let weights = [[1.0, 0.5, -0.2], [-1.0, 2.0, 0.1]];
        let bias = [1.0, -0.5];

        let fc = FullyConnected::new(weights, bias);
        let input = PixelArray::new([2.0, 3.0, -1.0]);
        let result = fc.forward(&input);

        assert_eq!(result.into_inner(), [4.7, 3.4]);
    }

    #[test]
    fn softmax() {
        let result = super::softmax(&PixelArray::new([
            2.1, 0.8, 3.2, 1.5, 0.3, 2.8, 1.1, 0.6, 1.9, 2.4,
        ]));

        assert_eq!(
            result.into_inner(),
            [
                0.102415346,
                0.027911441,
                0.30767277,
                0.05620674,
                0.016929146,
                0.20623921,
                0.037676506,
                0.022851957,
                0.08385061,
                0.13824628
            ]
        );
    }
}
