use std::marker::PhantomData;

use crate::image::Image;
use crate::layer::{BackPropagation, Layer};

#[derive(Debug, Clone)]
pub struct Flatten<Data, const LEN: usize> {
    pub data: Data,
    pub input: Image,
    _len: PhantomData<[f32; LEN]>,
}

pub trait FlattenData
where
    Self: Sized,
{
    fn flatten<const LEN: usize>(self) -> Flatten<Self, LEN>;
}

impl<T> FlattenData for T
where
    T: Layer,
{
    fn flatten<const LEN: usize>(self) -> Flatten<Self, LEN> {
        Flatten {
            data: self,
            input: Image::default(),
            _len: PhantomData,
        }
    }
}

impl<Data, const LEN: usize> Layer for Flatten<Data, LEN>
where
    Data: Layer<Item = Image>,
{
    type Input = Data::Input;
    type Item = [f32; LEN];

    fn forward(&mut self, input: Self::Input) -> Self::Item {
        let image = self.data.forward(input);
        assert_eq!(image.pixels.len(), LEN);
        let result = image.pixels.as_slice().try_into().unwrap();
        self.input = image;
        result
    }
}

impl<T, const LEN: usize> BackPropagation for Flatten<T, LEN>
where
    T: Layer + BackPropagation<Gradient = Image>,
{
    type Gradient = [f32; LEN];

    fn backprop(&mut self, output_gradient: Self::Gradient) {
        let shape = self.input.shape();
        assert_eq!(shape.width * shape.height * shape.channels, LEN);
        self.data.backprop(Image {
            width: shape.width,
            height: shape.height,
            channels: shape.channels,
            pixels: output_gradient.to_vec(),
        });
    }

    fn learning_rate(&self) -> f32 {
        self.data.learning_rate()
    }
}
