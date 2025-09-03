use std::marker::PhantomData;

pub type Mat1d<const W: usize> = Mat3d<1, 1, W>;

pub type Mat2d<const H: usize, const W: usize> = Mat3d<1, H, W>;

#[derive(Debug, Clone, PartialEq)]
pub struct Mat3d<const C: usize, const H: usize, const W: usize> {
    data: Vec<f32>,
    _shape: PhantomData<[[[f32; W]; H]; C]>,
}

impl<const C: usize, const H: usize, const W: usize> Default for Mat3d<C, H, W> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const W: usize> std::ops::Index<usize> for Mat3d<1, 1, W> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        self.chw_ref(0, 0, index)
    }
}

impl<const W: usize> std::ops::IndexMut<usize> for Mat3d<1, 1, W> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.chw_mut(0, 0, index)
    }
}

impl<const C: usize, const H: usize, const W: usize> Mat3d<C, H, W> {
    pub fn zero() -> Self {
        Self {
            data: vec![0.0; C * H * W],
            _shape: PhantomData,
        }
    }

    #[track_caller]
    pub fn new(data: impl IntoIterator<Item = f32>) -> Self {
        let data = data.into_iter().collect::<Vec<_>>();
        verify("new", data.len(), C, H, W);
        Self {
            data,
            _shape: PhantomData,
        }
    }

    #[track_caller]
    pub fn from_vec(data: Vec<f32>) -> Self {
        verify("from_vec", data.len(), C, H, W);
        Self {
            data,
            _shape: PhantomData,
        }
    }

    pub fn pad<const PAD: usize>(&self) -> Mat3d<C, { H + 2 * PAD }, { W + 2 * PAD }> {
        let mut padded = Vec::new();
        for c in 0..C {
            let c = c * H * W;
            let hw = &self.data[c..c + H * W];

            padded.extend((0..PAD * (W + 2 * PAD)).map(|_| 0.0));
            for h in 0..H {
                padded.extend((0..PAD).map(|_| 0.0));
                for w in 0..W {
                    padded.push(hw[h * H + w]);
                }
                padded.extend((0..PAD).map(|_| 0.0));
            }
            padded.extend((0..PAD * (W + 2 * PAD)).map(|_| 0.0));
        }
        Mat3d::from_vec(padded)
    }

    #[track_caller]
    pub fn reshape<const RC: usize, const RH: usize, const RW: usize>(self) -> Mat3d<RC, RH, RW> {
        verify("reshape", self.data.len(), RC, RH, RW);
        Mat3d::from_vec(self.data)
    }

    pub fn into_inner(self) -> Vec<f32> {
        self.data
    }

    #[track_caller]
    pub fn chw(&self, c: usize, h: usize, w: usize) -> f32 {
        self.data[c * H * W + h * W + w]
    }

    #[track_caller]
    pub fn chw_ref(&self, c: usize, h: usize, w: usize) -> &f32 {
        &self.data[c * H * W + h * W + w]
    }

    #[track_caller]
    pub fn chw_mut(&mut self, c: usize, h: usize, w: usize) -> &mut f32 {
        &mut self.data[c * H * W + h * W + w]
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.data.iter()
    }

    pub fn iter_channels(&self, channel: usize) -> impl Iterator<Item = &f32> {
        let c = channel * H * W;
        self.data[c..c + H * W].iter()
    }

    pub fn iter_all_channels(&self) -> impl Iterator<Item = &f32> {
        (0..C).flat_map(|c| {
            let c = c * H * W;
            self.data[c..c + H * W].iter()
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.data.iter_mut()
    }

    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.data.as_mut_slice()
    }
}

#[track_caller]
fn verify(func: &str, len: usize, c: usize, h: usize, w: usize) {
    debug_assert_eq!(
        len,
        c * h * w,
        "Called `Mat3d::<{c}, {h}, {w}>::{}` with invalid \
                data len. Expected {}, got {}",
        func,
        c * h * w,
        len,
    );
    // skip formatting string, but still check in release
    assert_eq!(len, c * h * w);
}

#[cfg(test)]
mod test {
    use crate::Layer;

    use super::*;

    impl<const C: usize, const H: usize, const W: usize> Layer for Mat3d<C, H, W> {
        type Input = ();
        type Item = Self;

        fn input(&mut self, _: Self::Input) {
            panic!("`Mat3d::input` does nothing");
        }

        fn forward(&mut self) -> Self::Item {
            self.clone()
        }

        fn backprop(&mut self, _: Self::Item, _: f32) {
            panic!("`Mat3d::backprop` does nothing");
        }
    }

    #[test]
    fn chw() {
        #[rustfmt::skip]
        let mat = Mat3d::<1, 3, 3>::new(
            [
                1.0, 2.0, 3.0, 
                4.0, 5.0, 6.0, 
                7.0, 8.0, 9.0,
            ]
        );
        assert_eq!(mat.chw(0, 0, 0), 1.0);
        assert_eq!(mat.chw(0, 0, 1), 2.0);
        assert_eq!(mat.chw(0, 0, 2), 3.0);
        assert_eq!(mat.chw(0, 1, 0), 4.0);
        assert_eq!(mat.chw(0, 1, 1), 5.0);
        assert_eq!(mat.chw(0, 1, 2), 6.0);
        assert_eq!(mat.chw(0, 2, 0), 7.0);
        assert_eq!(mat.chw(0, 2, 1), 8.0);
        assert_eq!(mat.chw(0, 2, 2), 9.0);

        #[rustfmt::skip]
        let mat = Mat3d::<3, 3, 1>::new(
            [
                1.0, 2.0, 3.0, 
                4.0, 5.0, 6.0, 
                7.0, 8.0, 9.0,
            ]
        );
        assert_eq!(mat.chw(0, 0, 0), 1.0);
        assert_eq!(mat.chw(0, 1, 0), 2.0);
        assert_eq!(mat.chw(0, 2, 0), 3.0);
        assert_eq!(mat.chw(1, 0, 0), 4.0);
        assert_eq!(mat.chw(1, 1, 0), 5.0);
        assert_eq!(mat.chw(1, 2, 0), 6.0);
        assert_eq!(mat.chw(2, 0, 0), 7.0);
        assert_eq!(mat.chw(2, 1, 0), 8.0);
        assert_eq!(mat.chw(2, 2, 0), 9.0);
    }

    #[test]
    fn pad() {
        #[rustfmt::skip]
        let input = Mat3d::<1, 2, 2>::new([
            1.0, 2.0,
            3.0, 4.0,
        ]);
        let padded = input.pad::<1>();
        #[rustfmt::skip]
        assert_eq!(padded.as_slice(), &[
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 0.0,
            0.0, 3.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]);

        #[rustfmt::skip]
        let input = Mat3d::<2, 2, 2>::new([
            // c1
            1.0, 2.0,
            3.0, 4.0,
            // c2
            1.0, 2.0,
            3.0, 4.0,
        ]);
        let padded = input.pad::<1>();
        #[rustfmt::skip]
        assert_eq!(padded.as_slice(), &[
            // c1
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 0.0,
            0.0, 3.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            // c2
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 0.0,
            0.0, 3.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]);
    }
}
