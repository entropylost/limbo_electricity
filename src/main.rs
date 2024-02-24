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
const MAX_CHARGE: f32 = 5.0;
const MAX_POTENTIAL: f32 = 5.0;

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

    let charges = device.create_tex2d::<f32>(PixelStorage::Float1, GRID_SIZE, GRID_SIZE, 1);
    let potential = device.create_tex2d::<f32>(PixelStorage::Float1, GRID_SIZE, GRID_SIZE, 1);

    let draw_kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos / SCALING;
            let c = charges.read(pos);
            let p = potential.read(pos);
            let color: Expr<Vec3<f32>> = if c != 0.0 {
                // TODO: Scale color by charge.
                if c > 0.0 {
                    Vec3::expr(1.0, 0.0, 0.0)
                } else {
                    Vec3::expr(0.0, 0.0, 1.0)
                }
                // *(c / MAX_CHARGE as f32)
            } else {
                Vec3::splat_expr(1.0) * ((p / MAX_POTENTIAL as f32) * 0.5 + 0.5)
            };
            display.write(display_pos, color.extend(1.0));
        }),
    );

    let sample_kernel = Kernel::<fn(u32)>::new(
        &device,
        &track!(|t| {
            let pos = dispatch_id().xy();
            let angle = rand_f32(pos, t, 0) * 2.0 * PI;
            let dir = Vec2::expr(angle.cos(), angle.sin());
            let dist = 10.0; // rand
            let target = pos.cast_f32() + (dir * 10.0);
            let target = (target + Vec2::splat(0.5)).cast_i32();
            if target.x < 0
                || target.x >= GRID_SIZE as i32
                || target.y < 0
                || target.y >= GRID_SIZE as i32
            {
                return;
            }
            let target = target.cast_u32();
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
                                propegate_potentials_kernel
                                    .dispatch_async([GRID_SIZE, GRID_SIZE, 1], &t),
                                activate_kernel.dispatch_async([GRID_SIZE, GRID_SIZE, 1]),
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
