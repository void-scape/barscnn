use std::cell::Cell;

use bevy::ecs::system::SystemId;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy_egui::egui::load::SizedTexture;
use bevy_egui::egui::{
    CollapsingHeader, Color32, ColorImage, CornerRadius, Stroke, TextureHandle, TextureOptions,
    Widget, emath,
};
use bevy_egui::{EguiContext, PrimaryEguiContext};
use bevy_egui::{EguiPrimaryContextPass, egui};
use cnn::Cnn;
use cnn::activation::{Activation, ActivationFunction};
use cnn::fc::FullyConnected;
use cnn::feature::FeatureMap;
use cnn::flatten::Flatten;
use cnn::matrix::{Mat1d, Mat3d};
use cnn::pool::MaxPool;
use cnn::softmax::Softmax;
use egui_plot::{Plot, PlotPoint, Points};

use crate::training::{Accuracy, Epoch, Evaluation, IterationsPerEpoch, LetterCnn, Loss, Network};

pub fn plugin(app: &mut App) {
    app.init_resource::<Textures>()
        .init_resource::<EvalPoints>()
        .add_plugins((
            bevy_egui::EguiPlugin::default(),
            bevy_inspector_egui::DefaultInspectorConfigPlugin,
        ))
        .add_systems(EguiPrimaryContextPass, (viz_update, viz_evaluation));
}

#[derive(Component)]
pub struct VizUpdateSystem(pub SystemId);

impl VizUpdateSystem {
    pub fn new<N>(commands: &mut Commands) -> Self
    where
        N: LetterCnn,
    {
        Self(commands.register_system(network_egui::<N>))
    }
}

fn viz_update(mut commands: Commands, update: Query<&VizUpdateSystem>) {
    for update in update.iter() {
        commands.run_system(update.0);
    }
}

#[derive(Default, Resource)]
struct EvalPoints {
    acc: Vec<PlotPoint>,
    epoch_acc: usize,
    epoch_min_acc: f32,
    epoch_max_acc: f32,

    loss: Vec<PlotPoint>,
    epoch_loss: usize,
    epoch_min_loss: f32,
    epoch_max_loss: f32,
}

fn viz_evaluation(world: &mut World) {
    let Ok(egui_context) = world
        .query_filtered::<&mut EguiContext, With<PrimaryEguiContext>>()
        .single(world)
    else {
        return;
    };
    let mut egui_context = egui_context.clone();

    let (loss, loss_epoch) = world
        .query_filtered::<(Ref<Loss>, &Epoch), With<Evaluation>>()
        .single(world)
        .unwrap();
    let loss_is_changed = loss.is_changed();
    let loss = *loss;
    let loss_epoch = loss_epoch.0;

    let (acc, acc_epoch) = world
        .query_filtered::<(Ref<Accuracy>, &Epoch), With<Evaluation>>()
        .single(world)
        .unwrap();
    let acc_is_changed = acc.is_changed();
    let acc = *acc;
    let acc_epoch = acc_epoch.0;

    let iter_per_epoch = world.resource::<IterationsPerEpoch>().0;

    let mut points = world.resource_mut::<EvalPoints>();
    if loss_is_changed {
        if points.epoch_loss != loss_epoch {
            points.epoch_loss = loss_epoch;
            points.epoch_min_loss = f32::MAX;
            points.epoch_max_loss = f32::MIN;
        }

        if loss.tick < points.epoch_min_loss {
            points.epoch_min_loss = loss.tick;
        }
        if loss.tick > points.epoch_max_loss {
            points.epoch_max_loss = loss.tick;
        }

        points.loss.push(PlotPoint::new(
            (loss.iteration + loss_epoch * iter_per_epoch) as f64,
            loss.tick,
        ));
    }

    if acc_is_changed {
        if points.epoch_acc != acc_epoch {
            points.epoch_acc = acc_epoch;
            points.epoch_min_acc = f32::MAX;
            points.epoch_max_acc = f32::MIN;
        }

        let p = acc.compute_percentage();
        if p < points.epoch_min_acc {
            points.epoch_min_acc = p;
        }
        if p > points.epoch_max_acc {
            points.epoch_max_acc = p;
        }

        points.acc.push(PlotPoint::new(
            (acc.iteration + acc_epoch * iter_per_epoch) as f64,
            acc.compute(),
        ));
    }

    egui::Window::new("Evaluation").show(egui_context.get_mut(), |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            Plot::new("eval").allow_scroll(false).show(ui, |ui| {
                ui.points(Points::new("loss", points.loss.as_slice()).color(Color32::BLUE));
                ui.points(Points::new("acc", points.acc.as_slice()).color(Color32::RED));
            });

            CollapsingHeader::new("Statistics")
                .default_open(true)
                .show(ui, |ui| {
                    egui::Grid::new("softmax_stats").show(ui, |ui| {
                        ui.label("Epoch Max Accuracy:");
                        ui.label(format!("{:.4}", points.epoch_max_acc));
                        ui.end_row();

                        ui.label("Epoch Min Accuracy:");
                        ui.label(format!("{:.4}", points.epoch_min_acc));
                        ui.end_row();

                        ui.label("Epoch Avg Accuracy:");
                        ui.label(format!("{:.2}%", acc.compute_percentage()));
                        ui.end_row();

                        ui.label("Epoch Max Loss:");
                        ui.label(format!("{:.4}", points.epoch_max_loss));
                        ui.end_row();

                        ui.label("Epoch Min Loss:");
                        ui.label(format!("{:.4}", points.epoch_min_loss));
                        ui.end_row();

                        ui.label("Epoch Avg Loss:");
                        ui.label(format!("{:.4}", loss.compute()));
                        ui.end_row();
                    });
                });
        });
    });
}

fn network_egui<N: LetterCnn>(world: &mut World) {
    let Ok(egui_context) = world
        .query_filtered::<&mut EguiContext, With<PrimaryEguiContext>>()
        .single(world)
    else {
        return;
    };
    let mut egui_context = egui_context.clone();

    let name = world
        .query_filtered::<&Name, With<Network<N>>>()
        .single(world)
        .unwrap();
    egui::Window::new(name.as_str()).show(egui_context.get_mut(), |ui| {
        egui::ScrollArea::both().show(ui, |ui| {
            let mut textures = world.remove_resource::<Textures>().unwrap();
            let mut value = world.query::<&mut Network<N>>().single_mut(world).unwrap();
            value.0.ui(
                ui,
                &mut textures,
                LayerId {
                    layer: 0,
                    cnn_typename: std::any::type_name::<N>(),
                },
            );
            textures.1 = false;
            world.insert_resource(textures);
        });
    });
}

pub trait InspectorUi {
    fn ui(&mut self, ui: &mut egui::Ui, textures: &mut Textures, id: LayerId);
}

#[derive(Default, Resource)]
pub struct Textures(HashMap<(LayerId, usize), TextureHandle>, bool);

impl Textures {
    pub fn force_update(&mut self) {
        self.1 = true;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerId {
    layer: usize,
    cnn_typename: &'static str,
}

impl LayerId {
    pub fn inc(self) -> Self {
        Self {
            layer: self.layer + 1,
            cnn_typename: self.cnn_typename,
        }
    }
}

trait IntoImage {
    fn into_image(self) -> cnn::image::Image;
}

impl<const H: usize, const W: usize> IntoImage for Mat3d<1, H, W> {
    fn into_image(self) -> cnn::image::Image {
        cnn::image::Image {
            width: W,
            height: H,
            channels: 1,
            pixels: self.into_inner(),
        }
    }
}

impl<const H: usize, const W: usize> InspectorUi for Cnn<1, H, W> {
    fn ui(&mut self, ui: &mut egui::Ui, textures: &mut Textures, id: LayerId) {
        ui.vertical(|ui| {
            ui.heading("Image Input Layer");
            ui.label(format!("Image: {}x{}x{}", 1, H, W));
            display_image(
                ui,
                textures,
                || color_image_direct(&self.0.clone().into_image()),
                id,
                0,
                emath::Vec2::ONE * 4.0,
                false,
            );
        });
    }
}

impl<T, const S: usize, const C: usize, const H: usize, const W: usize, const L: usize> InspectorUi
    for FeatureMap<T, S, C, H, W, L>
where
    T: InspectorUi,
{
    fn ui(&mut self, ui: &mut egui::Ui, textures: &mut Textures, id: LayerId) {
        ui.vertical(|ui| {
            CollapsingHeader::new("Inner Layer")
                .default_open(true)
                .show(ui, |ui| {
                    self.layer.ui(ui, textures, id.inc());
                });

            ui.separator();
            ui.heading("Feature Map Layer");
            ui.label(format!("{0}x{1}x{2} -> {0}x{1}x{3}", C, H, W, L));

            CollapsingHeader::new("Filters")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label(format!("Filter Weight Matrix: {}x{}x{}", C, S, S));
                    display_images(
                        ui,
                        textures,
                        || create_feature_map_filter_images(self),
                        L * C,
                        id,
                        0,
                        emath::Vec2::ONE * 32.0,
                        false,
                    );
                });
        });
    }
}

fn create_feature_map_filter_images<
    T,
    const S: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const L: usize,
>(
    fm: &FeatureMap<T, S, C, H, W, L>,
) -> Vec<cnn::image::Image> {
    fm.filters()
        .iter()
        .flat_map(|filter| {
            (0..C).map(|i| {
                let min = *filter
                    .weights
                    .iter_channels(i)
                    .min_by(|a, b| a.total_cmp(b))
                    .unwrap();
                let max = *filter
                    .weights
                    .iter_channels(i)
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap();

                cnn::image::Image {
                    width: S,
                    height: S,
                    channels: 1,
                    pixels: filter
                        .weights
                        .iter_channels(i)
                        .map(|p| (p / (max - min + f32::EPSILON)).clamp(0.0, 1.0))
                        .collect(),
                }
            })
        })
        .collect()
}

impl<T, F, const C: usize, const H: usize, const W: usize> InspectorUi for Activation<T, F, C, H, W>
where
    T: InspectorUi,
    F: ActivationFunction,
{
    fn ui(&mut self, ui: &mut egui::Ui, textures: &mut Textures, id: LayerId) {
        ui.vertical(|ui| {
            CollapsingHeader::new("Inner Layer")
                .default_open(true)
                .show(ui, |ui| {
                    self.layer.ui(ui, textures, id.inc());
                });

            ui.separator();
            ui.heading("Activation Layer");
            ui.label(format!(
                "{0} {1}x{2}x{3} -> {1}x{2}x{3}",
                std::any::type_name::<F>(),
                C,
                H,
                W
            ));

            CollapsingHeader::new("Input")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label("Image Threshold:");
                    thread_local! {
                        static THRESHOLD: Cell<(f32, f32)> = Cell::new((-1.0, 1.0));
                    }
                    let (mut lower, mut upper) = THRESHOLD.with(|t| t.get());
                    let mut force_update = false;
                    if egui_double_slider::DoubleSlider::new(&mut lower, &mut upper, -1.0..=1.0)
                        .separation_distance(0.1)
                        .width(250.0)
                        .scroll_factor(0.0)
                        .ui(ui)
                        .changed()
                    {
                        THRESHOLD.with(|t| t.set((lower, upper)));
                        force_update = true;
                    }

                    ui.label(format!("Input Matrix: {}x{}x{}", C, H, W));
                    let threshold = THRESHOLD.with(|t| t.get());
                    display_images(
                        ui,
                        textures,
                        || create_act_input_images(self.layer_input(), threshold.0, threshold.1),
                        C,
                        id,
                        0,
                        emath::Vec2::ONE * 4.0,
                        force_update,
                    );
                });

            ui.separator();

            CollapsingHeader::new("Activation Map")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label(format!("Activation Matrix: {}x{}x{}", C, H, W));
                    display_images(
                        ui,
                        textures,
                        || create_activation_images::<F, _, _, _>(self.layer_input()),
                        C,
                        id,
                        C,
                        emath::Vec2::ONE * 4.0,
                        false,
                    );
                });
        });
    }
}

fn create_act_input_images<const C: usize, const H: usize, const W: usize>(
    input: &Mat3d<C, H, W>,
    lower: f32,
    upper: f32,
) -> Vec<cnn::image::Image> {
    let range = upper - lower;
    (0..C)
        .map(|i| cnn::image::Image {
            width: W,
            height: H,
            channels: 1,
            pixels: input
                .iter_channels(i)
                .map(|p| p.clamp(lower, upper) / range)
                .collect(),
        })
        .collect()
}

fn create_activation_images<F, const C: usize, const H: usize, const W: usize>(
    input: &Mat3d<C, H, W>,
) -> Vec<cnn::image::Image>
where
    F: ActivationFunction,
{
    let mask = cnn::activation::activation(input, &mut F::pass_mask);
    (0..C)
        .map(|i| cnn::image::Image {
            width: W,
            height: H,
            channels: 1,
            pixels: mask.iter_channels(i).copied().collect(),
        })
        .collect()
}

impl<T, const SIZE: usize, const C: usize, const H: usize, const W: usize> InspectorUi
    for MaxPool<T, SIZE, C, H, W>
where
    T: InspectorUi,
    [(); H / SIZE]:,
    [(); W / SIZE]:,
{
    fn ui(&mut self, ui: &mut egui::Ui, textures: &mut Textures, id: LayerId) {
        ui.vertical(|ui| {
            CollapsingHeader::new("Inner Layer")
                .default_open(true)
                .show(ui, |ui| {
                    self.layer.ui(ui, textures, id.inc());
                });

            ui.separator();
            ui.heading("Max Pool Layer");
            ui.label(format!(
                "{}x{}x{} -> {}x{}x{}",
                C,
                H,
                W,
                C,
                H / SIZE,
                W / SIZE,
            ));

            CollapsingHeader::new("Input")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label("Image Threshold:");
                    thread_local! {
                        static THRESHOLD: Cell<(f32, f32)> = Cell::new((0.0, 1.0));
                    }
                    let (mut lower, mut upper) = THRESHOLD.with(|t| t.get());
                    let mut force_update = false;
                    if egui_double_slider::DoubleSlider::new(&mut lower, &mut upper, -1.0..=1.0)
                        .separation_distance(0.1)
                        .width(250.0)
                        .scroll_factor(0.0)
                        .ui(ui)
                        .changed()
                    {
                        THRESHOLD.with(|t| t.set((lower, upper)));
                        force_update = true;
                    }

                    ui.label(format!("Input Matrix: {}x{}x{}", C, H, W));
                    let threshold = THRESHOLD.with(|t| t.get());
                    display_images(
                        ui,
                        textures,
                        || create_max_pool_input_images(self, threshold.0, threshold.1),
                        C,
                        id,
                        0,
                        emath::Vec2::ONE * 4.0,
                        force_update,
                    );
                });

            ui.separator();

            CollapsingHeader::new("Output")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label("Image Threshold:");
                    thread_local! {
                        static THRESHOLD: Cell<(f32, f32)> = Cell::new((0.0, 1.0));
                    }
                    let (mut lower, mut upper) = THRESHOLD.with(|t| t.get());
                    let mut force_update = false;
                    if egui_double_slider::DoubleSlider::new(&mut lower, &mut upper, -1.0..=1.0)
                        .separation_distance(0.1)
                        .width(250.0)
                        .scroll_factor(0.0)
                        .ui(ui)
                        .changed()
                    {
                        THRESHOLD.with(|t| t.set((lower, upper)));
                        force_update = true;
                    }

                    ui.label(format!("Output Matrix: {}x{}x{}", C, H / SIZE, W / SIZE,));
                    let threshold = THRESHOLD.with(|t| t.get());
                    display_images(
                        ui,
                        textures,
                        || create_max_pool_output_images(self, threshold.0, threshold.1),
                        C,
                        id,
                        C,
                        emath::Vec2::ONE * 8.0,
                        force_update,
                    );
                });

            ui.separator();

            // ui.collapsing("Statistics", |ui| {
            //     let output_image = self.max_pool();
            //
            //     ui.label("Dimensional Statistics:");
            //     egui::Grid::new("dimension_stats").show(ui, |ui| {
            //         ui.label("Pool size:");
            //         ui.label(format!("{}×{}", SIZE, SIZE));
            //         ui.end_row();
            //
            //         ui.label("Input pixels:");
            //         ui.label(format!("{}", H * W));
            //         ui.end_row();
            //
            //         ui.label("Output pixels:");
            //         ui.label(format!("{}", (H / SIZE) * (W / SIZE)));
            //         ui.end_row();
            //
            //         let reduction_factor = (H * W) as f32 / ((H / SIZE) * (W / SIZE)) as f32;
            //         ui.label("Reduction factor:");
            //         ui.label(format!("{:.2}× fewer pixels", reduction_factor));
            //         ui.end_row();
            //
            //         let compression_ratio = (1.0 - 1.0 / reduction_factor) * 100.0;
            //         ui.label("Compression ratio:");
            //         ui.label(format!("{:.1}% reduction", compression_ratio));
            //         ui.end_row();
            //     });
            //
            //     #[derive(Debug)]
            //     struct ImageStats {
            //         mean: f32,
            //         std_dev: f32,
            //         min: f32,
            //         max: f32,
            //     }
            //
            //     #[derive(Debug)]
            //     struct PoolingStats {
            //         sparsity: f32,
            //         zero_count: usize,
            //         concentration: f32,
            //         edge_preservation: f32,
            //     }
            //
            //     fn calculate_image_stats(image: &cnn::image::Image) -> ImageStats {
            //         let pixels = &image.pixels;
            //         let len = pixels.len() as f32;
            //
            //         let mean = pixels.iter().sum::<f32>() / len;
            //         let variance = pixels.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / len;
            //         let std_dev = variance.sqrt();
            //         let min = pixels.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            //         let max = pixels.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            //
            //         ImageStats {
            //             mean,
            //             std_dev,
            //             min,
            //             max,
            //         }
            //     }
            //
            //     fn calculate_pooling_stats(
            //         input: &cnn::image::Image,
            //         output: &cnn::image::Image,
            //         pool_size: usize,
            //     ) -> PoolingStats {
            //         let zero_count = output.pixels.iter().filter(|&&x| x.abs() < 1e-6).count();
            //         let sparsity = zero_count as f32 / output.pixels.len() as f32;
            //
            //         // Calculate concentration (how much values cluster around the mean)
            //         let mean = output.pixels.iter().sum::<f32>() / output.pixels.len() as f32;
            //         let concentration = {
            //             let deviations: Vec<f32> =
            //                 output.pixels.iter().map(|&x| (x - mean).abs()).collect();
            //             let mean_deviation =
            //                 deviations.iter().sum::<f32>() / deviations.len() as f32;
            //             if mean_deviation == 0.0 {
            //                 1.0
            //             } else {
            //                 1.0 / (1.0 + mean_deviation)
            //             }
            //         };
            //
            //         // Simple edge preservation estimate (compare gradients)
            //         let edge_preservation = estimate_edge_preservation(input, output, pool_size);
            //
            //         PoolingStats {
            //             sparsity,
            //             zero_count,
            //             concentration,
            //             edge_preservation,
            //         }
            //     }
            //
            //     fn estimate_edge_preservation(
            //         input: &cnn::image::Image,
            //         output: &cnn::image::Image,
            //         pool_size: usize,
            //     ) -> f32 {
            //         // Simple gradient magnitude comparison
            //         let input_gradient_sum = calculate_gradient_magnitude(input);
            //         let output_gradient_sum = calculate_gradient_magnitude(output);
            //
            //         // Scale by the expected reduction due to pooling
            //         let expected_scale = 1.0 / (pool_size as f32).powi(2);
            //         let scaled_output_gradient = output_gradient_sum / expected_scale;
            //
            //         if input_gradient_sum == 0.0 {
            //             1.0
            //         } else {
            //             (scaled_output_gradient / input_gradient_sum).min(1.0)
            //         }
            //     }
            //
            //     fn calculate_gradient_magnitude(image: &cnn::image::Image) -> f32 {
            //         let mut total_gradient = 0.0;
            //
            //         for y in 0..image.height.saturating_sub(1) {
            //             for x in 0..image.width.saturating_sub(1) {
            //                 for c in 0..image.channels {
            //                     let idx = (y * image.width + x) * image.channels + c;
            //                     let idx_right = (y * image.width + x + 1) * image.channels + c;
            //                     let idx_down = ((y + 1) * image.width + x) * image.channels + c;
            //
            //                     let dx = image.pixels[idx_right] - image.pixels[idx];
            //                     let dy = image.pixels[idx_down] - image.pixels[idx];
            //                     let gradient_mag = (dx * dx + dy * dy).sqrt();
            //                     total_gradient += gradient_mag;
            //                 }
            //             }
            //         }
            //
            //         total_gradient
            //     }
            //
            //     ui.separator();
            //
            //     ui.label("Comparisons:");
            //     // let input_stats = calculate_image_stats(&self.input);
            //     // let output_stats = calculate_image_stats(&output_image);
            //
            //     egui::Grid::new("comparison_stats").show(ui, |ui| {
            //         ui.label("");
            //         ui.label("Input");
            //         ui.label("Output");
            //         ui.label("Change");
            //         ui.end_row();
            //
            //         ui.label("Mean:");
            //         ui.label(format!("{:.4}", input_stats.mean));
            //         ui.label(format!("{:.4}", output_stats.mean));
            //         let mean_change =
            //             ((output_stats.mean - input_stats.mean) / input_stats.mean) * 100.0;
            //         ui.label(format!("{:+.1}%", mean_change));
            //         ui.end_row();
            //
            //         ui.label("Std Dev:");
            //         ui.label(format!("{:.4}", input_stats.std_dev));
            //         ui.label(format!("{:.4}", output_stats.std_dev));
            //         let std_change = ((output_stats.std_dev - input_stats.std_dev)
            //             / input_stats.std_dev)
            //             * 100.0;
            //         ui.label(format!("{:+.1}%", std_change));
            //         ui.end_row();
            //
            //         ui.label("Min:");
            //         ui.label(format!("{:.4}", input_stats.min));
            //         ui.label(format!("{:.4}", output_stats.min));
            //         ui.label(format!("{:+.4}", output_stats.min - input_stats.min));
            //         ui.end_row();
            //
            //         ui.label("Max:");
            //         ui.label(format!("{:.4}", input_stats.max));
            //         ui.label(format!("{:.4}", output_stats.max));
            //         ui.label(format!("{:+.4}", output_stats.max - input_stats.max));
            //         ui.end_row();
            //
            //         ui.label("Range:");
            //         let input_range = input_stats.max - input_stats.min;
            //         let output_range = output_stats.max - output_stats.min;
            //         ui.label(format!("{:.4}", input_range));
            //         ui.label(format!("{:.4}", output_range));
            //         ui.label(format!("{:+.4}", output_range - input_range));
            //         ui.end_row();
            //     });
            //
            //     ui.separator();
            //
            //     ui.label("Pooling Analysis:");
            //
            //     let pooling_stats = calculate_pooling_stats(&self.input, &output_image, self.size);
            //
            //     egui::Grid::new("pooling_stats").show(ui, |ui| {
            //         ui.label("Output sparsity:");
            //         ui.label(format!(
            //             "{:.2}% ({} zeros)",
            //             pooling_stats.sparsity * 100.0,
            //             pooling_stats.zero_count
            //         ));
            //         ui.end_row();
            //
            //         ui.label("Value concentration:");
            //         ui.label(format!("{:.3}", pooling_stats.concentration));
            //         ui.end_row();
            //
            //         ui.label("Edge preservation:");
            //         ui.label(format!("{:.2}%", pooling_stats.edge_preservation * 100.0));
            //         ui.end_row();
            //     });
            // });
        });
    }
}

fn create_max_pool_input_images<
    T,
    const SIZE: usize,
    const C: usize,
    const H: usize,
    const W: usize,
>(
    max_pool: &MaxPool<T, SIZE, C, H, W>,
    lower: f32,
    upper: f32,
) -> Vec<cnn::image::Image> {
    let range = upper - lower;
    (0..C)
        .map(|i| cnn::image::Image {
            width: W,
            height: H,
            channels: 1,
            pixels: max_pool
                .layer_input()
                .iter_channels(i)
                .map(|p| p.clamp(lower, upper) / range)
                .collect(),
        })
        .collect()
}

fn create_max_pool_output_images<
    T,
    const SIZE: usize,
    const C: usize,
    const H: usize,
    const W: usize,
>(
    max_pool: &MaxPool<T, SIZE, C, H, W>,
    lower: f32,
    upper: f32,
) -> Vec<cnn::image::Image>
where
    [(); H / SIZE]:,
    [(); W / SIZE]:,
{
    let output = max_pool.max_pool();

    let range = upper - lower;
    (0..C)
        .map(|i| cnn::image::Image {
            width: W / SIZE,
            height: H / SIZE,
            channels: 1,
            pixels: output
                .iter_channels(i)
                .map(|p| p.clamp(lower, upper) / range)
                .collect(),
        })
        .collect()
}

impl<T, const C: usize, const H: usize, const W: usize> InspectorUi for Flatten<T, C, H, W>
where
    T: InspectorUi,
{
    fn ui(&mut self, ui: &mut egui::Ui, textures: &mut Textures, id: LayerId) {
        ui.vertical(|ui| {
            CollapsingHeader::new("Inner Layer")
                .default_open(true)
                .show(ui, |ui| {
                    self.layer.ui(ui, textures, id.inc());
                });

            ui.separator();
            ui.heading("Flatten Layer");
            ui.label(format!("{}x{}x{} -> {}x{}", C, H, W, 1, C * H * W,));
        });
    }
}

impl<T, const H: usize, const W: usize> InspectorUi for FullyConnected<T, H, W>
where
    T: InspectorUi,
{
    fn ui(&mut self, ui: &mut egui::Ui, textures: &mut Textures, id: LayerId) {
        ui.vertical(|ui| {
            CollapsingHeader::new("Inner Layer")
                .default_open(true)
                .show(ui, |ui| {
                    self.layer.ui(ui, textures, id.inc());
                });

            ui.separator();
            ui.heading("Fully Connected Layer");
            ui.label(format!("{}x{} -> {}x{}", 1, H, 1, W));

            ui.separator();

            CollapsingHeader::new("Weights")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label(format!("Weight Matrix: {}x{}", H, W));
                    display_image(
                        ui,
                        textures,
                        || create_weight_image(self),
                        id,
                        0,
                        emath::Vec2::ONE,
                        false,
                    );

                    ui.add_space(10.0);
                    let weights_flat: Vec<f32> = self.fc.weights.iter().copied().collect();
                    display_weight_stats(ui, &weights_flat, "Weight");
                });

            ui.separator();

            CollapsingHeader::new("Bias")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label(format!("Bias Vector: {}", W));
                    display_image(
                        ui,
                        textures,
                        || create_bias_image(self),
                        id,
                        1,
                        emath::Vec2::ONE * 20.0,
                        false,
                    );

                    ui.add_space(10.0);
                    display_weight_stats(ui, self.fc.bias.as_slice(), "Bias");
                });

            ui.separator();

            ui.collapsing("Layer Statistics", |ui| {
                let weights_flat: Vec<f32> = self.fc.weights.iter().copied().collect();

                let total_params = weights_flat.len() + W;
                let weight_magnitude = weights_flat.iter().map(|&w| w * w).sum::<f32>().sqrt();
                let bias_magnitude = self.fc.bias.iter().map(|&b| b * b).sum::<f32>().sqrt();

                egui::Grid::new("fc_stats").show(ui, |ui| {
                    ui.label("Total parameters:");
                    ui.label(format!("{}", total_params));
                    ui.end_row();

                    ui.label("Weight parameters:");
                    ui.label(format!("{}", weights_flat.len()));
                    ui.end_row();

                    ui.label("Bias parameters:");
                    ui.label(format!("{}", W));
                    ui.end_row();

                    ui.label("Weight L2 norm:");
                    ui.label(format!("{:.6}", weight_magnitude));
                    ui.end_row();

                    ui.label("Bias L2 norm:");
                    ui.label(format!("{:.6}", bias_magnitude));
                    ui.end_row();
                });
            });
        });
    }
}

fn display_weight_stats(ui: &mut egui::Ui, values: &[f32], label: &str) {
    if values.is_empty() {
        return;
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std_dev = variance.sqrt();
    let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    let zero_count = values.iter().filter(|&&x| x.abs() < 1e-6).count();
    let sparsity = zero_count as f32 / values.len() as f32;

    egui::Grid::new(format!("{}_stats", label.to_lowercase())).show(ui, |ui| {
        ui.label("Mean:");
        ui.label(format!("{:.6}", mean));
        ui.end_row();

        ui.label("Std deviation:");
        ui.label(format!("{:.6}", std_dev));
        ui.end_row();

        ui.label("Range:");
        ui.label(format!("[{:.4}, {:.4}]", min_val, max_val));
        ui.end_row();

        ui.label("Span:");
        ui.label(format!("{:.6}", range));
        ui.end_row();

        ui.label("Sparsity:");
        ui.label(format!("{:.2}% ({} zeros)", sparsity * 100.0, zero_count));
        ui.end_row();
    });
}

fn create_weight_image<T, const H: usize, const W: usize>(
    fc: &FullyConnected<T, H, W>,
) -> ColorImage {
    let min = *fc.fc.weights.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max = *fc.fc.weights.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    ColorImage::new(
        [H, W],
        fc.fc
            .weights
            .iter()
            .map(|&weight| {
                let v =
                    ((weight / (max - min + f32::EPSILON)).clamp(0.0, 1.0) * u8::MAX as f32) as u8;
                Color32::from_rgb(v, v, v)
                // let rgb = weight_to_rgb(weight, min, max);
                // Color32::from_rgb(rgb[0], rgb[1], rgb[2])
            })
            .collect(),
    )
}

fn create_bias_image<T, const H: usize, const W: usize>(
    fc: &FullyConnected<T, H, W>,
) -> ColorImage {
    let min = *fc.fc.bias.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max = *fc.fc.bias.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    ColorImage::new(
        [W, 1],
        fc.fc
            .bias
            .iter()
            .map(|&weight| {
                let v =
                    ((weight / (max - min + f32::EPSILON)).clamp(0.0, 1.0) * u8::MAX as f32) as u8;
                Color32::from_rgb(v, v, v)
            })
            .collect(),
    )
}

impl<T, const W: usize> InspectorUi for Softmax<T, W>
where
    T: InspectorUi,
{
    fn ui(&mut self, ui: &mut egui::Ui, textures: &mut Textures, id: LayerId) {
        ui.vertical(|ui| {
            CollapsingHeader::new("Inner Layer")
                .default_open(true)
                .show(ui, |ui| {
                    self.layer.ui(ui, textures, id.inc());
                });

            ui.separator();
            ui.heading("Softmax Layer");
            ui.label(format!("{}x{} -> {}x{}", 1, W, 1, W));

            ui.separator();
            CollapsingHeader::new("Input")
                .default_open(true)
                .show(ui, |ui| {
                    draw_bar_chart(ui, self.layer_input(), Color32::LIGHT_BLUE, true);
                });

            let output = self.softmax();
            CollapsingHeader::new("Output")
                .default_open(true)
                .show(ui, |ui| {
                    draw_bar_chart(ui, &output, Color32::LIGHT_GREEN, false);
                });

            ui.separator();
            ui.collapsing("Statistics", |ui| {
                let input_sum: f32 = self.layer_input().iter().sum();
                let input_max = self
                    .layer_input()
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let input_min = self
                    .layer_input()
                    .iter()
                    .fold(f32::INFINITY, |a, &b| a.min(b));

                let output_sum: f32 = output.iter().sum();
                let entropy = -output
                    .iter()
                    .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
                    .sum::<f32>();
                let max_prob_idx = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                egui::Grid::new("softmax_stats").show(ui, |ui| {
                    ui.label("Input sum:");
                    ui.label(format!("{:.4}", input_sum));
                    ui.end_row();

                    ui.label("Input range:");
                    ui.label(format!("[{:.4}, {:.4}]", input_min, input_max));
                    ui.end_row();

                    ui.label("Output sum:");
                    ui.label(format!("{:.4}", output_sum));
                    ui.end_row();

                    ui.label("Entropy:");
                    ui.label(format!("{:.4}", entropy));
                    ui.end_row();

                    ui.label("Predicted class:");
                    ui.label(format!("{} (p={:.4})", max_prob_idx, output[max_prob_idx]));
                    ui.end_row();
                });
            });
        });
    }
}

fn draw_bar_chart<const W: usize>(
    ui: &mut egui::Ui,
    input: &Mat1d<W>,
    color: Color32,
    allow_negative: bool,
) {
    ui.add_space(30.0);

    let height = 100.0;
    let bar_width = 30.0;
    let spacing = 5.0;
    let total_width = W as f32 * (bar_width + spacing) - spacing;

    let (rect, _) =
        ui.allocate_exact_size(emath::Vec2::new(total_width, height), egui::Sense::hover());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();

        let min_val = if allow_negative {
            input.iter().fold(0.0f32, |acc, &x| acc.min(x))
        } else {
            0.0
        };
        let max_val = input.iter().fold(0.0f32, |acc, &x| acc.max(x));
        let range = max_val - min_val;

        if range > 0.0 {
            for (i, &value) in input.iter().enumerate() {
                let x = rect.min.x + i as f32 * (bar_width + spacing);
                let normalized = (value - min_val) / range;
                let bar_height = normalized * height;

                let bar_rect = if allow_negative && min_val < 0.0 {
                    let baseline_y = rect.max.y - ((0.0 - min_val) / range) * height;
                    if value >= 0.0 {
                        emath::Rect::from_min_size(
                            emath::Pos2::new(
                                x,
                                baseline_y - bar_height + ((0.0 - min_val) / range) * height,
                            ),
                            emath::Vec2::new(
                                bar_width,
                                bar_height - ((0.0 - min_val) / range) * height,
                            ),
                        )
                    } else {
                        emath::Rect::from_min_size(
                            emath::Pos2::new(x, baseline_y),
                            emath::Vec2::new(
                                bar_width,
                                -bar_height + ((0.0 - min_val) / range) * height,
                            ),
                        )
                    }
                } else {
                    emath::Rect::from_min_size(
                        emath::Pos2::new(x, rect.max.y - bar_height),
                        emath::Vec2::new(bar_width, bar_height),
                    )
                };

                painter.rect_filled(bar_rect, CornerRadius::same(2), color);
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, Color32::WHITE),
                    egui::StrokeKind::Inside,
                );

                let text_pos = emath::Pos2::new(x + bar_width / 2.0, rect.min.y - 15.0);
                painter.text(
                    text_pos,
                    egui::Align2::CENTER_BOTTOM,
                    format!("{:.2}", value),
                    egui::FontId::proportional(10.0),
                    Color32::WHITE,
                );
            }
        }
    }
}

fn color_image_direct(image: &cnn::image::Image) -> ColorImage {
    assert_eq!(image.channels, 1);
    let pixels = image
        .pixels
        .iter()
        .map(|p| {
            // TODO: Negative in R channel positive in B channel?
            let v = (p.clamp(0.0, 1.0) * u8::MAX as f32) as u8;
            Color32::from_rgb(v, v, v)
        })
        .collect();
    ColorImage::new([image.width, image.height], pixels)
}

// fn weight_to_rgb(value: f32, min: f32, max: f32) -> [u8; 3] {
//     if max == min {
//         return [128, 128, 128];
//     }
//     let normalized = ((value - min) / (max - min)).clamp(0.0, 1.0);
//     hsv_to_rgb(normalized * 300.0, 1.0, 1.0)
// }
//
// fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
//     let h = h % 360.0;
//     let c = v * s;
//     let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
//     let m = v - c;
//
//     let (r_prime, g_prime, b_prime) = match h as i32 {
//         0..=59 => (c, x, 0.0),
//         60..=119 => (x, c, 0.0),
//         120..=179 => (0.0, c, x),
//         180..=239 => (0.0, x, c),
//         240..=299 => (x, 0.0, c),
//         _ => (c, 0.0, x),
//     };
//
//     [
//         ((r_prime + m) * 255.0) as u8,
//         ((g_prime + m) * 255.0) as u8,
//         ((b_prime + m) * 255.0) as u8,
//     ]
// }

fn display_image(
    ui: &mut egui::Ui,
    textures: &mut Textures,
    mut image: impl FnMut() -> ColorImage,
    id: LayerId,
    index: usize,
    scale: emath::Vec2,
    force_update: bool,
) {
    let force_update = textures.1 || force_update;
    let handle = match (textures.0.get(&(id, index)), force_update) {
        (Some(handle), false) => handle.clone(),
        _ => {
            let color_image = image();
            let handle = ui
                .ctx()
                .load_texture("", color_image, TextureOptions::NEAREST);
            textures.0.insert((id, index), handle.clone());
            handle
        }
    };

    bevy_egui::egui::Image::new(SizedTexture {
        size: handle.size_vec2(),
        id: handle.id(),
    })
    .texture_options(TextureOptions::NEAREST)
    .fit_to_exact_size(handle.size_vec2() * scale)
    .ui(ui);
}

fn display_images(
    ui: &mut egui::Ui,
    textures: &mut Textures,
    mut images: impl FnMut() -> Vec<cnn::image::Image>,
    image_count: usize,
    id: LayerId,
    index: usize,
    scale: emath::Vec2,
    force_update: bool,
) {
    let force_update = textures.1 || force_update;

    let need_images =
        force_update || (0..image_count).any(|i| !textures.0.contains_key(&(id, index + i)));

    if !need_images {
        let cols = (image_count as f32).sqrt().ceil() as usize;
        let rows = (image_count + cols - 1) / cols;

        egui::Grid::new(format!("images_grid_{:?}_{}", id, index))
            .spacing([5.0, 5.0])
            .show(ui, |ui| {
                for row in 0..rows {
                    for col in 0..cols {
                        let img_index = row * cols + col;
                        if img_index < image_count {
                            if let Some(handle) = textures.0.get(&(id, index + img_index)) {
                                bevy_egui::egui::Image::new(SizedTexture {
                                    size: handle.size_vec2(),
                                    id: handle.id(),
                                })
                                .texture_options(TextureOptions::NEAREST)
                                .fit_to_exact_size(handle.size_vec2() * scale)
                                .ui(ui);
                            }
                        }
                    }
                    ui.end_row();
                }
            });
        return;
    }

    let images_vec = images();
    let actual_count = images_vec.len().min(image_count);
    assert_eq!(actual_count, image_count);

    let cols = (actual_count as f32).sqrt().ceil() as usize;
    let rows = (actual_count + cols - 1) / cols;

    egui::Grid::new(format!("images_grid_{:?}_{}", id, index))
        .spacing([5.0, 5.0])
        .show(ui, |ui| {
            for row in 0..rows {
                for col in 0..cols {
                    let img_index = row * cols + col;
                    if img_index < actual_count {
                        let image = &images_vec[img_index];

                        assert_eq!(image.channels, 1);
                        let pixels = image
                            .pixels
                            .iter()
                            .map(|p| {
                                let v = (p.clamp(0.0, 1.0) * u8::MAX as f32) as u8;
                                Color32::from_rgb(v, v, v)
                            })
                            .collect();
                        let color_image = ColorImage::new([image.width, image.height], pixels);
                        let handle =
                            ui.ctx()
                                .load_texture("", color_image, TextureOptions::NEAREST);

                        textures.0.insert((id, index + img_index), handle.clone());
                        bevy_egui::egui::Image::new(SizedTexture {
                            size: handle.size_vec2(),
                            id: handle.id(),
                        })
                        .texture_options(TextureOptions::NEAREST)
                        .fit_to_exact_size(handle.size_vec2() * scale)
                        .ui(ui);
                    }
                }
                ui.end_row();
            }
        });
}
