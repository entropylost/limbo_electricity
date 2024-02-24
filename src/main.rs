use std::{
    collections::HashSet,
    env::current_exe,
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
fn rand(pos: Expr<Vec2<u32>>, t: Expr<u32>) -> Expr<u32> {
    let input = t + pos.x * GRID_SIZE + pos.y * GRID_SIZE * GRID_SIZE;
    hash(input)
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

    let statics = device.create_tex2d::<u32>(PixelStorage::Byte1, GRID_SIZE, GRID_SIZE, 1);

    let alpha_a = device.create_tex2d::<u32>(PixelStorage::Byte1, GRID_SIZE, GRID_SIZE, 1);
    let alpha_b = device.create_tex2d::<u32>(PixelStorage::Byte1, GRID_SIZE, GRID_SIZE, 1);

    let draw_kernel = Kernel::<fn(Tex2d<u32>)>::new(
        &device,
        &track!(|alpha| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos / SCALING;
            let alpha = alpha.read(pos);
            let color: Expr<Vec3<f32>> = if alpha == 1 {
                Vec3::expr(1.0, 0.0, 0.0)
            } else if statics.read(pos) == 1 {
                Vec3::splat_expr(0.3)
            } else {
                Vec3::splat_expr(0.0)
            };
            display.write(display_pos, color.extend(1.0));
        }),
    );

    let update_kernel = Kernel::<fn(Tex2d<u32>, Tex2d<u32>, u32)>::new(
        &device,
        &track!(|alpha, next_alpha, t| {
            let pos = dispatch_id().xy() + 1;
            let alpha = alpha.read(pos);
            let r = rand(pos, t);
            if alpha == 1 {
                next_alpha.write(pos - Vec2::expr(0, 1), 1);
                statics.write(pos, 1);
            }
            if alpha == 2 || alpha == 1 {
                if r >= u32::MAX / 10 {
                    // Left
                    next_alpha.write(pos + Vec2::expr(1, 0), 2);
                }
                statics.write(pos, 1);
            }
            if alpha == 3 || alpha == 1 {
                if r >= u32::MAX / 10 {
                    // Right
                    next_alpha.write(pos - Vec2::expr(1, 0), 3);
                }
                statics.write(pos, 1);
            }
        }),
    );

    let clear_kernel = Kernel::<fn(Tex2d<u32>)>::new(
        &device,
        &track!(|alpha| {
            let pos = dispatch_id().xy();
            alpha.write(pos, 0);
        }),
    );

    let update_alpha_kernel = Kernel::<fn(Tex2d<u32>, Vec2<u32>, u32)>::new(
        &device,
        &track!(|alpha, pos, value| {
            alpha.write(pos, value);
        }),
    );

    let mut parity = false;

    let mut cursor_pos = PhysicalPosition::new(0.0, 0.0);

    let mut active_buttons = HashSet::new();

    let mut update_cursor = |active_buttons: &HashSet<MouseButton>,
                             cursor_pos: PhysicalPosition<f64>,
                             alpha: &Tex2d<u32>| {
        let pos = Vec2::new(
            (cursor_pos.x as u32) / SCALING,
            (cursor_pos.y as u32) / SCALING,
        );
        if active_buttons.contains(&MouseButton::Left) {
            update_alpha_kernel.dispatch([1, 1, 1], alpha, &pos, &1);
        }
    };
    let update_cursor = &mut update_cursor;

    let mut t = 0;

    let start = Instant::now();

    let dt = Duration::from_secs_f64(1.0 / 60.0);

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
                        let alpha = if parity { &alpha_a } else { &alpha_b };
                        let next_alpha = if parity { &alpha_b } else { &alpha_a };
                        parity = !parity;
                        t += 1;
                        // update_cursor(&active_buttons, cursor_pos, alpha);
                        {
                            let commands = vec![
                                clear_kernel.dispatch_async([GRID_SIZE, GRID_SIZE, 1], next_alpha),
                                update_kernel.dispatch_async(
                                    [GRID_SIZE - 2, GRID_SIZE - 2, 1],
                                    alpha,
                                    next_alpha,
                                    &t,
                                ),
                                draw_kernel.dispatch_async(
                                    [GRID_SIZE * SCALING, GRID_SIZE * SCALING, 1],
                                    next_alpha,
                                ),
                            ];
                            scope.submit(commands);
                        }
                    }
                    window.request_redraw();
                }
                WindowEvent::CursorMoved { position, .. } => {
                    cursor_pos = position;
                    let alpha = if parity { &alpha_a } else { &alpha_b };
                    // update_cursor(&active_buttons, cursor_pos, alpha);
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
                    let alpha = if parity { &alpha_a } else { &alpha_b };
                    update_cursor(&active_buttons, cursor_pos, alpha);
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
