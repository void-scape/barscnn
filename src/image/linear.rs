use std::marker::PhantomData;

use super::pixel::PixelArray;

#[derive(Debug)]
pub struct Reduce<const INPUT: usize, const OUTPUT: usize> {
    weights: [[f32; INPUT]; OUTPUT],
    bias: [f32; OUTPUT],
}

impl<const INPUT: usize, const OUTPUT: usize> Reduce<INPUT, OUTPUT> {
    pub fn new(weights: [[f32; INPUT]; OUTPUT], bias: [f32; OUTPUT]) -> Self {
        Self { weights, bias }
    }

    pub fn forward(&self, pixels: &PixelArray<INPUT>) -> PixelArray<OUTPUT> {
        forward(self, pixels.as_slice())
    }
}

fn forward<const INPUT: usize, const OUTPUT: usize>(
    reduce: &Reduce<INPUT, OUTPUT>,
    pixels: &[f32],
) -> PixelArray<OUTPUT> {
    debug_assert_eq!(pixels.len(), INPUT);

    let mut output = [0.0; OUTPUT];
    for r in 0..OUTPUT {
        let mut result = 0.0;
        for c in 0..INPUT {
            result += reduce.weights[r][c] * pixels[c];
        }
        result += reduce.bias[r];
        output[r] = result;
    }
    PixelArray::new(output)
}

#[derive(Debug, Default)]
pub struct Softmax<const OUTPUT: usize>(PhantomData<[f32; OUTPUT]>);

pub fn softmax<const OUTPUT: usize>(pixels: &PixelArray<OUTPUT>) -> PixelArray<OUTPUT> {
    debug_assert_eq!(pixels.len(), OUTPUT);

    let pixels = pixels.into_inner();
    let exp = pixels.map(|v| v.exp());
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

        let reduce = Reduce::new(weights, bias);
        let input = PixelArray::new([2.0, 3.0, -1.0]);
        let result = reduce.forward(&input);

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
