use crate::Layer;
use crate::matrix::Mat3d;

#[derive(Debug, Clone)]
pub struct MaxPool<T, const SIZE: usize, const C: usize, const H: usize, const W: usize> {
    pub layer: T,
    input: Mat3d<C, H, W>,
}

impl<T, const SIZE: usize, const C: usize, const H: usize, const W: usize>
    MaxPool<T, SIZE, C, H, W>
{
    #[doc(hidden)]
    const NON_ZERO_SIZE: () = assert!(SIZE != 0, "`SIZE` must be greater than 0");

    #[doc(hidden)]
    const SIZE_CONSTRAINTS: () = assert!(
        H >= SIZE && W >= SIZE,
        "`H` and `W` must be larger than or equal to `SIZE`"
    );

    pub fn max_pool(&self) -> Mat3d<C, { H / SIZE }, { W / SIZE }> {
        max_pool::<SIZE, _, _, _>(&self.input)
    }

    pub fn max_unpool(&self, matrix: Mat3d<C, { H / SIZE }, { W / SIZE }>) -> Mat3d<C, H, W> {
        max_unpool::<SIZE, _, _, _>(&self.input, &matrix)
    }

    pub fn layer_input(&self) -> &Mat3d<C, H, W> {
        &self.input
    }
}

pub trait MaxPoolLayer<const C: usize, const H: usize, const W: usize>
where
    Self: Sized,
{
    fn max_pool_layer<const SIZE: usize>(self) -> MaxPool<Self, SIZE, C, H, W>;
}

impl<T, const C: usize, const H: usize, const W: usize> MaxPoolLayer<C, H, W> for T
where
    T: Layer,
{
    fn max_pool_layer<const SIZE: usize>(self) -> MaxPool<Self, SIZE, C, H, W> {
        let _invalid_pool_size = MaxPool::<Self, SIZE, C, H, W>::NON_ZERO_SIZE;
        let _invalid_pool_size = MaxPool::<Self, SIZE, C, H, W>::SIZE_CONSTRAINTS;
        MaxPool {
            layer: self,
            input: Mat3d::zero(),
        }
    }
}

impl<T, const SIZE: usize, const C: usize, const H: usize, const W: usize> Layer
    for MaxPool<T, SIZE, C, H, W>
where
    T: Layer<Item = Mat3d<C, H, W>>,
    [(); H / SIZE]:,
    [(); W / SIZE]:,
{
    type Input = T::Input;
    type Item = Mat3d<C, { H / SIZE }, { W / SIZE }>;

    fn input(&mut self, input: Self::Input) {
        self.layer.input(input);
    }

    fn forward(&mut self) -> Self::Item {
        self.input = self.layer.forward();
        self.max_pool()
    }

    fn backprop(&mut self, output_gradient: Self::Item, learning_rate: f32) {
        self.layer
            .backprop(self.max_unpool(output_gradient), learning_rate);
    }
}

pub fn max_pool<const SIZE: usize, const C: usize, const H: usize, const W: usize>(
    input: &Mat3d<C, H, W>,
) -> Mat3d<C, { H / SIZE }, { W / SIZE }> {
    debug_assert!(SIZE != 0);
    debug_assert!(SIZE <= W && SIZE <= H);

    let width = W / SIZE;
    let height = H / SIZE;
    let channels = C;
    let mut output = Mat3d::<C, { H / SIZE }, { W / SIZE }>::zero();

    for mh in 0..height {
        for mw in 0..width {
            for c in 0..channels {
                let mut max = f32::MIN;
                for h in 0..SIZE {
                    for w in 0..SIZE {
                        let pixel = input.chw(c, mh * SIZE + h, mw * SIZE + w);
                        max = pixel.max(max);
                    }
                }
                *output.chw_mut(c, mh, mw) = max;
            }
        }
    }

    output
}

fn max_unpool<const SIZE: usize, const C: usize, const H: usize, const W: usize>(
    original: &Mat3d<C, H, W>,
    pooled: &Mat3d<C, { H / SIZE }, { W / SIZE }>,
) -> Mat3d<C, H, W> {
    debug_assert!(SIZE != 0);
    debug_assert!(SIZE <= W && SIZE <= H);

    let mut output = Mat3d::zero();
    for mh in 0..H / SIZE {
        for mw in 0..W / SIZE {
            for c in 0..C {
                let mut max = f32::MIN;
                let mut i = (0, 0, 0);
                for h in 0..SIZE {
                    for w in 0..SIZE {
                        let h = mh * SIZE + h;
                        let w = mw * SIZE + w;

                        let pixel = original.chw(c, h, w);
                        if pixel > max {
                            max = pixel;
                            i = (c, h, w);
                        }
                    }
                }
                *output.chw_mut(i.0, i.1, i.2) = pooled.chw(c, mh, mw);
            }
        }
    }
    output
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn fuzz() {
        // TODO: Const matrices make this kind of robustness check hard.
        max_pool::<1, _, _, _>(&Mat3d::<2, 3, 6>::zero());
        max_pool::<2, _, _, _>(&Mat3d::<1, 9, 2>::zero());
        max_pool::<3, _, _, _>(&Mat3d::<2, 293, 4>::zero());
        max_pool::<4, _, _, _>(&Mat3d::<1, 4, 4>::zero());
    }

    #[test]
    fn max_pool_5x3_2() {
        #[rustfmt::skip]
        let input = Mat3d::<1, 3, 5>::new([
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.2, 0.3, 0.4, 0.5, 0.6,
            0.3, 0.4, 0.5, 0.6, 0.7,
        ]);
        let result = max_pool::<2, _, _, _>(&input);
        assert_eq!(result.as_slice(), &[0.3, 0.5]);

        let result = max_unpool::<2, _, _, _>(&input, &result);
        #[rustfmt::skip]
        assert_eq!(
            result.as_slice(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.3, 0.0, 0.5, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );
    }
}
