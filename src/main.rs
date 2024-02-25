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
const MAX_FIELD: f32 = 5.0;

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
    let field = device.create_tex2d::<Vec2<f32>>(PixelStorage::Float2, GRID_SIZE, GRID_SIZE, 1);
    let field_deltas =
        device.create_tex3d::<Vec2<f32>>(PixelStorage::Float2, GRID_SIZE, GRID_SIZE, 3, 1);

    let particles = device.create_tex2d::<u32>(PixelStorage::Byte1, GRID_SIZE, GRID_SIZE, 1);
    let particle_velocity =
        device.create_tex2d::<Vec2<f32>>(PixelStorage::Float2, GRID_SIZE, GRID_SIZE, 1);

    let draw_kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos / SCALING;
            let c = charges.read(pos);
            let p = particles.read(pos);
            let color: Expr<Vec3<f32>> = if p == 1 {
                Vec3::expr(1.0, 1.0, 1.0)
            } else if c != 0.0 {
                if c > 0.0 {
                    Vec3::expr(1.0, 0.0, 0.0)
                } else {
                    Vec3::expr(0.0, 0.0, 1.0)
                }
            } else {
                let f = field.read(pos);
                if !f.x.is_finite() || !f.y.is_finite() {
                    Vec3::expr(1.0, 0.0, 1.0)
                } else {
                    (f / (MAX_FIELD * 2.0) + 0.5).extend(0.0)
                }
            };
            display.write(display_pos, color.extend(1.0));
        }),
    );

    // Grid is staggered.
    // +---------------+
    // |      p.y      |
    // |               |
    // | p.x           |
    // |               |
    // |               |
    // +---------------+

    let solve_divergence = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let pos = dispatch_id().xy();
            let charge = charges.read(pos);
            let f = field.read(pos);
            let l = f.x;
            let t = f.y;
            let r = field.read(pos + Vec2::expr(1, 0)).x;
            let b = field.read(pos + Vec2::expr(0, 1)).y;
            let target_divergence = charge * 1.0;
            let divergence = r + b - l - t;
            let delta = (divergence - target_divergence) / 4.0;
            field_deltas.write(pos.extend(0), Vec2::expr(delta, delta));
            field_deltas.write((pos + Vec2::expr(1, 0)).extend(1), Vec2::expr(-delta, 0.0));
            field_deltas.write((pos + Vec2::expr(0, 1)).extend(2), Vec2::expr(0.0, -delta));
        }),
    );
    let apply_deltas = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let pos = dispatch_id().xy();
            let delta = field_deltas.read(pos.extend(0))
                + field_deltas.read(pos.extend(1))
                + field_deltas.read(pos.extend(2));
            field.write(pos, field.read(pos) + delta);
            field_deltas.write(pos.extend(0), Vec2::splat_expr(0.0));
            field_deltas.write(pos.extend(1), Vec2::splat_expr(0.0));
            field_deltas.write(pos.extend(2), Vec2::splat_expr(0.0));
        }),
    );

    let update_kernel = Kernel::<fn(f32, u32)>::new(
        &device,
        &track!(|dt, t| {
            // TODO: This should probably move the velocity instead.
            let pos = dispatch_id().xy();
            // Also make this gather not scatter.
            let p = particles.read(pos);
            if p == 1 {
                let vel = particle_velocity.read(pos) + field.read(pos) * 1.0 / 30.0;
                let movement = vel * dt;
                let sign = (movement > 0.0).cast::<i32>() * 2 - 1;
                let abs = movement.abs();
                let int = abs.floor();
                let frac = abs - int;
                let int = int.cast_i32();
                let abs = (Vec2::expr(rand_f32(pos, t, 0), rand_f32(pos, t, 1)) < frac)
                    .cast::<i32>()
                    + int;
                let new_pos = pos.cast_i32() + abs * sign;
                if new_pos.x < 0
                    || new_pos.x >= GRID_SIZE as i32
                    || new_pos.y < 0
                    || new_pos.y >= GRID_SIZE as i32
                {
                    return;
                }
                let new_pos = new_pos.cast_u32();
                if (new_pos != pos).any() {
                    particles.write(new_pos, 1);
                    particle_velocity.write(new_pos, vel);
                    particles.write(pos, 0);
                } else {
                    particle_velocity.write(pos, vel);
                }
            }
        }),
    );

    let write_charge_kernel = Kernel::<fn(Vec2<u32>, f32)>::new(
        &device,
        &track!(|pos, value| {
            charges.write(pos, value);
        }),
    );
    let write_particle_kernel = Kernel::<fn(Vec2<u32>)>::new(
        &device,
        &track!(|pos| {
            particles.write(pos, 1);
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
            write_charge_kernel.dispatch([1, 1, 1], &pos, &-5.0);
        }
        if active_buttons.contains(&MouseButton::Right) {
            write_charge_kernel.dispatch([1, 1, 1], &pos, &5.0);
        }
        if active_buttons.contains(&MouseButton::Middle) {
            write_particle_kernel.dispatch([1, 1, 1], &pos);
        }
    };
    let update_cursor = &mut update_cursor;

    let mut t = 0;

    let start = Instant::now();

    let dt = Duration::from_secs_f64(1.0 / 60.0);
    let step = 1.0;

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
                                solve_divergence.dispatch_async([GRID_SIZE - 1, GRID_SIZE - 1, 1]),
                                apply_deltas.dispatch_async([GRID_SIZE, GRID_SIZE, 1]),
                                update_kernel.dispatch_async([GRID_SIZE, GRID_SIZE, 1], &step, &t),
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
