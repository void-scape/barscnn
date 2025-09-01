use std::marker::PhantomData;
use std::time::Duration;

use bevy::ecs::component::HookContext;
use bevy::ecs::system::SystemId;
use bevy::ecs::world::DeferredWorld;
use bevy::prelude::*;
use bevy_prng::WyRand;
use bevy_rand::prelude::*;
use cnn::linear::Softmax;
use cnn::prelude::*;
use rand::Rng;

use crate::AlphaState;
use crate::viz::{InspectorUi, Textures, VizUpdateSystem};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, SystemSet)]
pub struct TrainSet;

pub fn plugin(app: &mut App) {
    app.init_resource::<IterationsPerEpoch>()
        .add_systems(Startup, training_data)
        .add_systems(
            Update,
            (
                run_network_systems,
                epoch,
                (loss, accuracy),
                debug_results,
                force_texture_update,
            )
                // .run_if(run_at_freq(1.0))
                .run_if(run_at_freq(1.0 / 30.0))
                .chain()
                .in_set(TrainSet)
                .run_if(in_state(AlphaState::Train)),
        );
}

fn run_at_freq(freq: f32) -> impl Fn(Res<Time>, Local<Timer>) -> bool {
    move |time, mut timer| {
        timer.tick(time.delta());
        let result = timer.just_finished();
        if result {
            timer.set_duration(Duration::from_secs_f32(freq));
            timer.set_mode(TimerMode::Repeating);
        }
        result
    }
}

#[derive(Component)]
pub struct Training;

#[derive(Component)]
pub struct Validation;

#[derive(Component)]
pub struct Evaluation;

pub trait LetterCnn:
    Send
    + Sync
    + InspectorUi
    + Layer<Input = cnn::image::Image, Item = [f32; 26]>
    + BackPropagation
    + BackPropagateLetter
    + 'static
{
}

impl<T> LetterCnn for T where
    T: Send
        + Sync
        + InspectorUi
        + Layer<Input = cnn::image::Image, Item = [f32; 26]>
        + BackPropagation
        + BackPropagateLetter
        + 'static
{
}

#[derive(Component)]
#[require(NetworkEntity, Entropy::<WyRand>::default())]
#[component(on_insert = Self::insert_system)]
pub struct Network<N: LetterCnn>(pub N);

impl<N: LetterCnn> Network<N> {
    fn insert_system(mut world: DeferredWorld, ctx: HookContext) {
        let training = world
            .commands()
            .register_system(forward::<Training, N, true>);
        let validation = world
            .commands()
            .register_system(forward::<Validation, N, false>);
        let evaluation = world
            .commands()
            .register_system(forward::<Evaluation, N, false>);

        let update = VizUpdateSystem::new::<N>(&mut world.commands());
        world.commands().entity(ctx.entity).insert((
            update,
            NetworkSystem::<Training>(training, PhantomData),
            NetworkSystem::<Validation>(validation, PhantomData),
            NetworkSystem::<Evaluation>(evaluation, PhantomData),
            children![
                (ForwardResult::default(), Training),
                (ForwardResult::default(), Validation),
                (ForwardResult::default(), Evaluation)
            ],
        ));
    }
}

#[derive(Default, Component)]
pub struct NetworkEntity;

#[derive(Component)]
pub struct NetworkSystem<Data>(SystemId, PhantomData<Data>);

fn run_network_systems(
    mut commands: Commands,
    systems: Query<(
        &NetworkSystem<Training>,
        &NetworkSystem<Validation>,
        &NetworkSystem<Evaluation>,
    )>,
) {
    for (training, validation, evaluation) in systems.iter() {
        commands.run_system(training.0);
        commands.run_system(validation.0);
        commands.run_system(evaluation.0);
    }
}

pub trait BackPropagateLetter {
    fn backprop_letter(&mut self, letter_index: usize, result: [f32; 26]);
}

impl<Data> BackPropagateLetter for Softmax<Data, 26>
where
    Softmax<Data, 26>: BackPropagation<Gradient = [f32; 26]>,
{
    fn backprop_letter(&mut self, letter_index: usize, result: [f32; 26]) {
        self.backprop_index(letter_index, result);
    }
}

fn forward<Data, N, const TRAIN: bool>(
    data: Single<&TrainingData, With<Data>>,
    network: Single<(&Children, &mut Entropy<WyRand>, &mut Network<N>)>,
    mut results: Query<&mut ForwardResult, With<Data>>,
) where
    Data: Component,
    N: LetterCnn,
{
    let (children, mut entropy, mut cnn) = network.into_inner();

    let training_index = entropy.random_range(0..data.0.len());
    let sample = data.0[training_index].clone();
    let result = cnn.0.forward(sample.image);
    if TRAIN {
        cnn.0.backprop_letter(sample.letter_index, result);
    }

    for entity in children.iter() {
        if let Ok(mut tresult) = results.get_mut(entity) {
            tresult.softmax = result;
            tresult.target = sample.letter_index;
            tresult.result = result
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap();
            return;
        }
    }
    panic!("Failed to find `ForwardResult`");
}

#[derive(Default, Component)]
#[require(Accuracy, Loss, Epoch)]
pub struct ForwardResult {
    pub target: usize,
    pub result: usize,
    pub softmax: [f32; 26],
}

#[derive(Default, Component)]
pub struct Accuracy {
    pub predicted: usize,
    pub iteration: usize,
}

impl Accuracy {
    pub fn compute(&self) -> f32 {
        self.predicted as f32 / self.iteration as f32
    }

    pub fn compute_percentage(&self) -> f32 {
        self.compute() * 100.0
    }
}

fn accuracy(mut networks: Query<(&ForwardResult, &mut Accuracy)>) {
    for (result, mut accuracy) in networks.iter_mut() {
        accuracy.predicted += (result.target == result.result) as usize;
        accuracy.iteration += 1;
    }
}

#[derive(Default, Component, Clone, Copy)]
pub struct Loss {
    pub tick: f32,
    pub accumulation: f32,
    pub iteration: usize,
}

impl Loss {
    pub fn compute(&self) -> f32 {
        self.accumulation / self.iteration as f32
    }
}

fn loss(mut networks: Query<(&ForwardResult, &mut Loss)>) {
    for (result, mut loss) in networks.iter_mut() {
        loss.tick = -result.softmax[result.result].ln();
        loss.accumulation += loss.tick;
        loss.iteration += 1;
    }
}

#[derive(Resource)]
pub struct IterationsPerEpoch(pub usize);

impl Default for IterationsPerEpoch {
    fn default() -> Self {
        Self(1_000)
    }
}

#[derive(Default, Component)]
pub struct Epoch(pub usize);

fn epoch(
    iter_per_epoch: Res<IterationsPerEpoch>,
    mut networks: Query<(&mut Accuracy, &mut Loss, &mut Epoch)>,
) {
    for (mut accuracy, mut loss, mut epoch) in networks.iter_mut() {
        assert_eq!(accuracy.iteration, loss.iteration);
        if accuracy.iteration > iter_per_epoch.0 {
            *accuracy = Accuracy::default();
            *loss = Loss::default();
            epoch.0 += 1;
        }
    }
}

#[derive(Component)]
pub struct TrainingData(pub Vec<Sample>);

#[derive(Clone)]
pub struct Sample {
    pub letter_index: usize,
    pub image: cnn::image::Image,
}

fn training_data(mut commands: Commands) {
    let mut len = 0;
    visit_dirs(
        std::path::Path::new("./data/letters"),
        &mut |entry: &std::fs::DirEntry| {
            if entry.path().extension().is_some_and(|ext| ext == "bmp") {
                len += 1;
            }
        },
    )
    .unwrap();

    let amt = 0.01;
    let progress = indicatif::ProgressBar::new((len as f32 * amt) as u64);
    std::thread::scope(|s| {
        let h1 = s.spawn(|| TrainingData(read_letters(&progress, 0.6 * amt, 0.0 * amt)));
        let h2 = s.spawn(|| TrainingData(read_letters(&progress, 0.2 * amt, 0.6 * amt)));
        let h3 = s.spawn(|| TrainingData(read_letters(&progress, 0.2 * amt, 0.8 * amt)));
        commands.spawn((h1.join().unwrap(), Training));
        commands.spawn((h2.join().unwrap(), Validation));
        commands.spawn((h3.join().unwrap(), Evaluation));
    });
    progress.finish();
    commands.set_state(AlphaState::Train);
}

fn visit_dirs(
    dir: &std::path::Path,
    cb: &mut impl FnMut(&std::fs::DirEntry),
) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, cb)?;
            } else {
                cb(&entry);
            }
        }
    }
    Ok(())
}

fn read_letters(progress: &indicatif::ProgressBar, percentage: f32, offset: f32) -> Vec<Sample> {
    let mut samples = Vec::new();
    for letter_index in 0..26 {
        let letter = ('A' as u8 + letter_index as u8) as char;
        let len = std::fs::read_dir(format!("./data/letters/{}", letter))
            .unwrap()
            .filter(|f| {
                f.as_ref()
                    .is_ok_and(|f| f.path().extension().is_some_and(|ext| ext == "bmp"))
            })
            .count();

        let start = len as f32 * offset;
        let end = (len as f32 - start) * percentage;
        let start = start.round() as usize;
        let end = end.round() as usize;

        for i in start..start + end {
            let path = format!("./data/letters/{}/{}-{}.bmp", letter, letter, i);
            let bytes = std::fs::read(path).unwrap();
            progress.inc(1);
            samples.push(Sample {
                letter_index,
                image: cnn::image::bmp::from_bytes(&bytes).unwrap(),
            });
        }
    }
    samples
}

fn debug_results(
    networks: Query<(&Name, &Children), With<NetworkEntity>>,
    training: Query<(&Accuracy, &Loss, &Epoch), With<Training>>,
    validation: Query<(&Accuracy, &Loss, &Epoch), With<Validation>>,
    evaluation: Query<(&Accuracy, &Loss, &Epoch), With<Evaluation>>,
) -> Result {
    for (name, children) in networks.iter() {
        for child in children.iter() {
            if let Some((data, (acc, loss, epoch))) =
                training.get(child).map(|al| ("T", al)).ok().or_else(|| {
                    validation
                        .get(child)
                        .ok()
                        .map(|al| ("Val", al))
                        .or_else(|| evaluation.get(child).map(|al| ("Eval", al)).ok())
                })
            {
                assert_eq!(loss.iteration, acc.iteration);
                debug!(
                    "[{}] {}\t| Epoch: {} | It: {} | L: {:.2} | Acc: {:.2}%",
                    name.as_str(),
                    data,
                    epoch.0,
                    loss.iteration,
                    loss.compute(),
                    acc.compute_percentage(),
                );
            }
        }
    }
    Ok(())
}

fn force_texture_update(mut textures: ResMut<Textures>) {
    textures.force_update();
}
