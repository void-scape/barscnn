use crate::Layer;
use crate::filter::Filter;
use crate::matrix::Mat3d;

#[derive(Debug, Clone)]
pub struct FeatureMap<
    T,
    const S: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const L: usize,
> {
    pub layer: T,
    filters: [Filter<S, C>; L],
    input: Mat3d<C, H, W>,
}

impl<T, const S: usize, const C: usize, const H: usize, const W: usize, const L: usize>
    FeatureMap<T, S, C, H, W, L>
{
    pub fn feature_map(&self) -> Mat3d<L, { H - S + 1 }, { W - S + 1 }> {
        crate::filter::conv(&self.input, &self.filters)
    }
}

pub trait FeatureMapLayer<
    const S: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const L: usize,
> where
    Self: Sized,
{
    fn feature_map_layer(self, filters: [Filter<S, C>; L]) -> FeatureMap<Self, S, C, H, W, L>;
}

impl<T, const S: usize, const C: usize, const H: usize, const W: usize, const L: usize>
    FeatureMapLayer<S, C, H, W, L> for T
{
    fn feature_map_layer(self, filters: [Filter<S, C>; L]) -> FeatureMap<Self, S, C, H, W, L> {
        FeatureMap {
            layer: self,
            filters,
            input: Mat3d::zero(),
        }
    }
}

impl<T, const S: usize, const C: usize, const H: usize, const W: usize, const L: usize> Layer
    for FeatureMap<T, S, C, H, W, L>
where
    T: Layer<Item = Mat3d<C, H, W>>,
    [(); H - S + 1]:,
    [(); W - S + 1]:,
{
    type Input = T::Input;
    type Item = Mat3d<L, { H - S + 1 }, { W - S + 1 }>;

    fn input(&mut self, input: Self::Input) {
        self.layer.input(input);
    }

    fn forward(&mut self) -> Self::Item {
        self.input = self.layer.forward();
        self.feature_map()
    }

    fn backprop(&mut self, output_gradient: Self::Item, learning_rate: f32) {
        self.layer.backprop(
            crate::filter::apply_filter_gradients(
                &mut self.filters,
                &self.input,
                &output_gradient,
                learning_rate,
            ),
            learning_rate,
        );
    }
}
