// use bevy::log::{Level, LogPlugin};
use bevy::prelude::*;
use bevy::window::WindowResolution;
use bevy_prng::WyRand;

use self::training::Network;

mod training;
mod viz;

const WIDTH: f32 = 1000.0;
const HEIGHT: f32 = 750.0;

fn main() {
    let seed: u64 = 234;

    App::default()
        .add_plugins((
            DefaultPlugins
                // .set(LogPlugin {
                //     level: Level::DEBUG,
                //     ..Default::default()
                // })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: WindowResolution::new(WIDTH, HEIGHT),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
            bevy_rand::plugin::EntropyPlugin::<WyRand>::with_seed(seed.to_ne_bytes()),
        ))
        .add_plugins((training::plugin, viz::plugin))
        .add_systems(Update, close_on_escape)
        .add_systems(Startup, spawn)
        .init_state::<AlphaState>()
        .run();
}

fn close_on_escape(mut writer: EventWriter<AppExit>, input: Res<ButtonInput<KeyCode>>) {
    if input.just_pressed(KeyCode::Escape) {
        writer.write(AppExit::Success);
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, States)]
pub enum AlphaState {
    #[default]
    LoadData,
    Train,
}

fn spawn(mut commands: Commands) {
    commands.spawn(Camera2d);

    use cnn::prelude::*;

    const FILT1: usize = 8;
    let filt1: [_; FILT1] = [
        // filter::uniform_3x3::<1>(); FILT1
        //
        filter::vertical_sobel::<1>(),
        filter::horizontal_sobel(),
        filter::gaussian_blur_3x3(),
        filter::laplacian_3x3(),
        filter::identity(),
        filter::uniform_3x3(),
        filter::vertical_sobel(),
        filter::horizontal_sobel(),
    ];
    // const FILT2: usize = 4;
    // let mut filt2 = [filter::identity(); FILT2];

    const INPUT: usize = 1568;
    let fc1 = FcWeights::<INPUT, 26>::glorot();
    // let mut fc2 = FcWeights::<_, 64>::glorot();
    // let mut fc3 = FcWeights::<_, 26>::glorot();

    let cnn = ImageCnn::learning_rate(0.001)
        .feature_map(filt1)
        .relu()
        .max_pool(2)
        // .feature_map(&mut filt2)
        // .relu()
        // .max_pool(2)
        .flatten()
        .fully_connected(fc1)
        .softmax();
    commands.spawn((Name::new("av1"), Network(cnn)));
}
