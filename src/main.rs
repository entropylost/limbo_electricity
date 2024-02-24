use std::{
    collections::HashSet,
    env::current_exe,
    f32::consts::PI,
    time::{Duration, Instant},
};

use luisa::{
    lang::types::vector::{Vec2, Vec3, Vec4},
    prelude::*,
};
use luisa_compute as luisa;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

const GRID_SIZE: u32 = 128;
const SCALING: u32 = 8;
const MAX_VOLTAGE: u32 = 16;

#[tracked]
fn hash(x: Expr<u32>) -> Expr<u32> {
    let x = x.var();
    *x ^= x >> 17;
    *x *= 0xed5ad4bb;
    *x ^= x >> 11;
    *x *= 0xac4c1b51;
    *x ^= x >> 15;
    *x *= 0x31848bab;
    *x ^= x >> 14;
    **x
}

#[tracked]
fn rand(pos: Expr<Vec2<u32>>, t: Expr<u32>, c: u32) -> Expr<u32> {
    let input = t
        + pos.x * GRID_SIZE
        + pos.y * GRID_SIZE * GRID_SIZE
        + c * GRID_SIZE * GRID_SIZE * GRID_SIZE;
    hash(input)
}

#[tracked]
fn rand_f32(pos: Expr<Vec2<u32>>, t: Expr<u32>, c: u32) -> Expr<f32> {
    rand(pos, t, c).as_f32() / u32::MAX as f32
}

#[tracked]
fn index(pos: Expr<Vec2<i32>>) -> Expr<u32> {
    if pos.x < 0 || pos.y < 0 || pos.x >= GRID_SIZE as i32 || pos.y >= GRID_SIZE as i32 {
        u32::MAX.expr()
    } else {
        (pos.x + pos.y * GRID_SIZE as i32).cast_u32()
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Value, PartialEq, Eq)]
struct Lightning {
    source: Vec2<u32>,
    potential: u32,
    active: bool,
    exists: bool,
    // Only exists if active.
    target: Vec2<u32>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Value, PartialEq, Eq)]
struct Voltage {
    voltage: u32,
    discharging: bool,
    exists: bool,
}

fn main() {
    luisa::init_logger();
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cuda");

    let event_loop = EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(
            GRID_SIZE * SCALING,
            GRID_SIZE * SCALING,
        ))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let swapchain = device.create_swapchain(
        &window,
        &device.default_stream(),
        GRID_SIZE * SCALING,
        GRID_SIZE * SCALING,
        false,
        false,
        3,
    );
    let display = device.create_tex2d::<Vec4<f32>>(
        swapchain.pixel_storage(),
        GRID_SIZE * SCALING,
        GRID_SIZE * SCALING,
        1,
    );

    let voltages = device.create_buffer::<Voltage>((GRID_SIZE * GRID_SIZE) as usize);
    let lightnings = device.create_buffer::<Lightning>((GRID_SIZE * GRID_SIZE) as usize);

    let draw_kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos / SCALING;
            let index = index(pos.cast_i32());
            let v = voltages.read(index);
            let l = lightnings.read(index);
            let color: Expr<Vec3<f32>> = if l.exists {
                if l.active {
                    Vec3::expr(1.0, 1.0, 1.0)
                } else {
                    Vec3::expr(0.0, l.potential.as_f32() / MAX_VOLTAGE as f32, 1.0)
                }
            } else if v.exists {
                Vec3::expr(1.0, v.voltage.as_f32() / MAX_VOLTAGE as f32, 0.0)
            } else {
                Vec3::expr(0.0, 0.0, 0.0)
            };
            display.write(display_pos, color.extend(1.0));
        }),
    );

    let update_kernel = Kernel::<fn(u32)>::new(
        &device,
        &track!(|t| {
            let pos = dispatch_id().xy();
            let pos_index = index(pos.cast_i32());
            let voltage = voltages.read(pos_index);
            let lightning = lightnings.read(pos_index);
            if voltage.exists && lightning.exists && voltage.voltage < lightning.potential {
                let lightning = lightning.var();
                // Valid destination.
                *lightning.active = true.expr();
                *lightning.target = pos;
                lightnings.write(pos_index, lightning);
            } else if lightning.active && lightning.exists {
                // Discharge time.
                // Activate source.
                let source_lightning = lightnings.read(index(lightning.source.cast_i32()));
                if source_lightning.exists && !source_lightning.active {
                    let source_lightning = source_lightning.var();
                    *source_lightning.active = true.expr();
                    *source_lightning.target = lightning.target;
                    lightnings.write(index(lightning.source.cast_i32()), source_lightning);
                }
                // Need double buffering here really.
            } else if voltage.exists || lightning.exists {
                // Activate possibly.
                let source_lightning = lightnings.read(index(lightning.source.cast_i32()));
                if source_lightning.exists && source_lightning.active {
                    let lightning = lightning.var();
                    *lightning.active = true.expr();
                    *lightning.target = source_lightning.target;
                    lightnings.write(pos_index, lightning);
                    // TODO: Necessary?
                    return;
                }

                // Decay self
                if lightning.exists {
                    // TODO: Interferes with above with return.
                    let lightning = lightning.var();
                    if lightning.potential == 0 {
                        *lightning.exists = false.expr();
                    } else if rand_f32(pos, t, 1) < 0.5 {
                        *lightning.potential = lightning.potential - 1;
                    }
                    lightnings.write(pos_index, lightning);
                }

                // Perform scan.
                let next_potential = if lightning.exists {
                    lightning.potential
                } else {
                    voltage.voltage
                };
                if next_potential == 0 {
                    return;
                }
                let next_potential = next_potential - 1;
                let angle = rand_f32(pos, t, 0) * 2.0 * PI;
                let dir = Vec2::expr(angle.cos(), angle.sin());
                let target = pos.cast_f32() + (dir * (1 << next_potential).cast_f32());
                let target = (target + Vec2::splat(0.5)).cast_i32();
                let target_index = index(target);
                if target_index == u32::MAX {
                    return;
                }
                let target_lightning = lightnings.read(target_index);
                if target_lightning.exists
                    && (target_lightning.active || target_lightning.potential > next_potential)
                {
                    // TODO: Maybe just rerout active so you still get a better path.
                    return;
                }
                let next_lightning = Lightning::from_comps_expr(LightningComps {
                    source: pos,
                    potential: next_potential,
                    active: false.expr(),
                    exists: true.expr(),
                    target: Vec2::splat_expr(0),
                });
                lightnings.write(target_index, next_lightning);
            }
        }),
    );

    let update_voltage_kernel = Kernel::<fn(Vec2<u32>, u32)>::new(
        &device,
        &track!(|pos, value| {
            voltages.write(
                index(pos.cast_i32()),
                Voltage::from_comps_expr(VoltageComps {
                    voltage: value,
                    discharging: false.expr(),
                    exists: true.expr(),
                }),
            );
        }),
    );

    let mut cursor_pos = PhysicalPosition::new(0.0, 0.0);

    let mut active_buttons = HashSet::new();

    let mut update_cursor = |active_buttons: &HashSet<MouseButton>,
                             cursor_pos: PhysicalPosition<f64>| {
        let pos = Vec2::new(
            (cursor_pos.x as u32) / SCALING,
            (cursor_pos.y as u32) / SCALING,
        );
        if active_buttons.contains(&MouseButton::Left) {
            update_voltage_kernel.dispatch([1, 1, 1], &pos, &0);
        }
        if active_buttons.contains(&MouseButton::Right) {
            update_voltage_kernel.dispatch([1, 1, 1], &pos, &5);
        }
    };
    let update_cursor = &mut update_cursor;

    let mut t = 0;

    let start = Instant::now();

    let dt = Duration::from_secs_f64(1.0 / 10.0);

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }
                WindowEvent::RedrawRequested => {
                    let scope = device.default_stream().scope();
                    scope.present(&swapchain, &display);

                    if dt * t < start.elapsed() {
                        t += 1;
                        // update_cursor(&active_buttons, cursor_pos);
                        {
                            let commands = vec![
                                update_kernel.dispatch_async([GRID_SIZE, GRID_SIZE, 1], &t),
                                draw_kernel.dispatch_async([
                                    GRID_SIZE * SCALING,
                                    GRID_SIZE * SCALING,
                                    1,
                                ]),
                            ];
                            scope.submit(commands);
                        }
                    }
                    window.request_redraw();
                }
                WindowEvent::CursorMoved { position, .. } => {
                    cursor_pos = position;
                    update_cursor(&active_buttons, cursor_pos);
                }
                WindowEvent::MouseInput { button, state, .. } => {
                    match state {
                        ElementState::Pressed => {
                            active_buttons.insert(button);
                        }
                        ElementState::Released => {
                            active_buttons.remove(&button);
                        }
                    }
                    update_cursor(&active_buttons, cursor_pos);
                }
                _ => (),
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => (),
        })
        .unwrap();
}
