use std::marker::PhantomData;

use super::Image;
use super::feature::FeatureSet;
use super::layer::{BackPropagation, Layer};
use super::pixel::{Grayscale, PixelArray, Pixels};

#[derive(Debug)]
pub struct Flatten<Data, const LEN: usize> {
    data: Data,
    shape: Shape,
    _len: PhantomData<[Data; LEN]>,
}

pub trait FlattenData<const LEN: usize>
where
    Self: Sized,
{
    fn flatten(self) -> Flatten<Self, LEN>;
}

impl<T, const LEN: usize> FlattenData<LEN> for T
where
    T: Layer,
    T::Item: Flattenable<LEN>,
{
    fn flatten(self) -> Flatten<Self, LEN> {
        Flatten {
            data: self,
            shape: Shape {
                width: 0,
                height: 0,
            },
            _len: PhantomData,
        }
    }
}

impl<T, const LEN: usize> Layer for Flatten<T, LEN>
where
    T: Layer,
    T::Item: Flattenable<LEN>,
{
    type Item = PixelArray<LEN>;

    fn forward(&mut self) -> Self::Item {
        let item = self.data.forward();
        self.shape = item.shape();
        item.flatten()
    }
}

impl<T, const LEN: usize> BackPropagation for Flatten<T, LEN>
where
    T: Layer + BackPropagation<Gradient = T::Item>,
    T::Item: Flattenable<LEN>,
{
    type Gradient = PixelArray<LEN>;

    fn backprop(&mut self, output_gradient: Self::Gradient, learning_rate: f32) {
        let gradient = T::Item::unflatten(output_gradient, self.shape);
        self.data.backprop(gradient, learning_rate);
    }
}

#[derive(Debug, Clone, Copy)]
struct Shape {
    width: u32,
    height: u32,
}

trait Flattenable<const LEN: usize>: Clone {
    fn shape(&self) -> Shape;
    fn flatten(&self) -> PixelArray<LEN>;
    fn unflatten(pixels: PixelArray<LEN>, shape: Shape) -> Self;
}

impl<const LEN: usize> Flattenable<LEN> for Image<Grayscale> {
    fn shape(&self) -> Shape {
        Shape {
            width: self.width,
            height: self.height,
        }
    }

    fn flatten(&self) -> PixelArray<LEN> {
        if self.pixels.len() != LEN {
            panic!(
                "called `flatten` with the incorrect `OUTPUT`, expected {}, got {LEN}",
                self.pixels.len()
            );
        }
        let mut arr = [0.0; LEN];
        arr.copy_from_slice(self.pixels.as_slice());
        PixelArray::new(arr)
    }

    fn unflatten(pixels: PixelArray<LEN>, shape: Shape) -> Self {
        Self {
            width: shape.width,
            height: shape.height,
            pixels: Pixels::new(pixels.into_inner().to_vec()),
        }
    }
}

impl<const LEN: usize, const DEPTH: usize> Flattenable<LEN> for FeatureSet<DEPTH> {
    fn shape(&self) -> Shape {
        debug_assert!(
            self.iter()
                .all(|img| img.width == self[0].width && img.height == self[0].height)
        );
        Shape {
            width: self[0].width,
            height: self[1].height,
        }
    }

    fn flatten(&self) -> PixelArray<LEN> {
        debug_assert_eq!(self.iter().map(|img| img.pixels.len()).sum::<usize>(), LEN);
        let mut arr = [0.0; LEN];
        let mut offset = 0;
        for img in self.iter() {
            arr[offset..offset + img.pixels.len()].copy_from_slice(img.pixels.as_slice());
            offset += img.pixels.len();
        }
        PixelArray::new(arr)
    }

    fn unflatten(pixels: PixelArray<LEN>, shape: Shape) -> Self {
        let mut set = std::array::from_fn(|_| Image {
            width: shape.width,
            height: shape.height,
            pixels: Pixels::new(Vec::new()),
        });

        let img_size = (shape.width * shape.height) as usize;
        for i in 0..DEPTH {
            set[i].pixels =
                Pixels::new(pixels.as_slice()[i * img_size..(i + 1) * img_size].to_vec());
        }

        FeatureSet::new(set)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn flatten() {
        let pixels = [1.0, 2.0, 3.0, 4.0];
        let img = Image {
            width: 2,
            height: 2,
            pixels: Pixels::new(pixels.to_vec()),
        };
        let flattened = <_ as Flattenable<4>>::flatten(&img).into_inner();

        assert_eq!(flattened, pixels);
        assert_eq!(
            <Image<Grayscale> as Flattenable<4>>::unflatten(
                PixelArray::new(flattened),
                Shape {
                    width: 2,
                    height: 2
                }
            )
            .pixels
            .as_slice(),
            img.pixels.as_slice(),
        );
    }
}
