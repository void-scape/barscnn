#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use bevy::prelude::*;
use bevy::window::WindowResolution;
use bevy_prng::WyRand;

use self::training::Network;

mod training;
mod viz;

const WIDTH: f32 = 1000.0;
const HEIGHT: f32 = 750.0;
const SEED: u64 = 123;

fn main() {
    App::default()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: WindowResolution::new(WIDTH, HEIGHT),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            bevy_rand::plugin::EntropyPlugin::<WyRand>::with_seed(SEED.to_ne_bytes()),
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

    let f1 = filter::feature_set::<12, 5, _>(|i| filter::xavier(SEED + i as u64));
    let f1_len = f1.len();
    let f2 = filter::feature_set::<24, 5, _>(|i| filter::xavier(SEED + i as u64 + f1_len as u64));

    let fc1 = FcWeights::glorot(SEED);
    let fc2 = FcWeights::glorot(SEED + 1);
    let fc3 = FcWeights::glorot(SEED + 2);

    let cnn = Cnn::<1, 32, 32>::default()
        .feature_map_layer(f1)
        .leaky_relu_layer()
        .max_pool_layer::<2>()
        .feature_map_layer(f2)
        .leaky_relu_layer()
        .max_pool_layer::<2>()
        .flatten_layer()
        .fully_connected_layer::<256>(fc1)
        .leaky_relu_layer()
        .fully_connected_layer::<128>(fc2)
        .leaky_relu_layer()
        .fully_connected_layer(fc3)
        .softmax_layer();
    commands.spawn((Name::new("av1"), Network(cnn)));
}
