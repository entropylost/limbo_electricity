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
    event::{ElementState, Event, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
};

// TODO: Make this 2047 or something so that mipmapping works.
const GRID_SIZE: u32 = 128;
const SCALING: u32 = 8;
const MAX_FIELD: f32 = 5.0;
const DOWNSCALE_COUNT: u32 = 7;
const CHARGE: f32 = 5.0;

#[derive(Debug, Clone, Copy, Value)]
#[repr(u32)]
pub enum View {
    Field = 0,
    Divergence = 1,
    Curl = 2,
    TrailOnly = 3,
}
impl View {
    fn next(&mut self) {
        *self = match self {
            Self::Field => Self::Divergence,
            Self::Divergence => Self::Curl,
            Self::Curl => Self::TrailOnly,
            Self::TrailOnly => Self::Field,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Runtime {
    cursor_pos: PhysicalPosition<f64>,
    t: u32,
    viewed_layer: u32,
    view: View,
    activate_multigrid: bool,
}
impl Default for Runtime {
    fn default() -> Self {
        Self {
            cursor_pos: PhysicalPosition::new(0.0, 0.0),
            t: 0,
            viewed_layer: 0,
            view: View::Field,
            activate_multigrid: true,
        }
    }
}

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
    let input = t + pos.x * GRID_SIZE + pos.y * GRID_SIZE * GRID_SIZE + c * 1063; //* GRID_SIZE * GRID_SIZE * GRID_SIZE;
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

    let charges = device.create_tex3d::<f32>(
        PixelStorage::Float1,
        GRID_SIZE,
        GRID_SIZE,
        DOWNSCALE_COUNT,
        1,
    );
    let magnets = device.create_tex3d::<f32>(
        PixelStorage::Float1,
        GRID_SIZE,
        GRID_SIZE,
        DOWNSCALE_COUNT,
        1,
    );
    let field = device.create_tex3d::<Vec2<f32>>(
        PixelStorage::Float2,
        GRID_SIZE + 1,
        GRID_SIZE + 1,
        DOWNSCALE_COUNT,
        1,
    );
    let divergence_error = device.create_tex3d::<f32>(
        PixelStorage::Float2,
        GRID_SIZE + 1,
        GRID_SIZE + 1,
        DOWNSCALE_COUNT,
        1,
    );
    let curl_error = device.create_tex3d::<f32>(
        PixelStorage::Float2,
        GRID_SIZE - 1,
        GRID_SIZE - 1,
        DOWNSCALE_COUNT,
        1,
    );

    let particles = device.create_tex2d::<u32>(PixelStorage::Byte1, GRID_SIZE, GRID_SIZE, 1);
    let particle_velocity =
        device.create_tex2d::<Vec2<f32>>(PixelStorage::Float2, GRID_SIZE, GRID_SIZE, 1);
    let trail = device.create_tex2d::<f32>(PixelStorage::Float1, GRID_SIZE, GRID_SIZE, 1);

    let draw_kernel = Kernel::<fn(u32, View)>::new(
        &device,
        &track!(|layer, view| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos / SCALING;
            let field_pos = (pos >> layer).extend(layer);
            let c = charges.read(field_pos) / (1 << (2 * layer)).as_f32();
            let p = particles.read(pos);
            let color: Expr<Vec3<f32>> = if p == 1 {
                Vec3::expr(1.0, 1.0, 1.0)
            } else if c != 0.0 {
                if c > 0.0 {
                    Vec3::expr(1.0, 0.0, 0.0) * c / CHARGE
                } else {
                    Vec3::expr(0.0, 0.0, 1.0) * (-c / CHARGE)
                }
            } else {
                if view.as_u32() == View::Curl.expr().as_u32() {
                    let c = curl_error.read(field_pos);
                    // TODO: Add color or something.
                    c.abs() * Vec3::splat(1.0)
                } else if view.as_u32() == View::Divergence.expr().as_u32() {
                    let d = divergence_error.read(field_pos);
                    if d > 0.0 {
                        Vec3::expr(0.0, 1.0, 0.0) * d
                    } else {
                        Vec3::expr(1.0, 0.0, 0.0) * (-d)
                    }
                } else if view.as_u32() == View::TrailOnly.expr().as_u32() {
                    let t = trail.read(pos);
                    Vec3::splat_expr(t)
                } else if view.as_u32() == View::Field.expr().as_u32() {
                    let f = field.read(field_pos) / (1 << layer).as_f32();
                    (f / (MAX_FIELD * 2.0) + 0.5).extend(0.0)
                } else {
                    Vec3::splat_expr(0.0)
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
    let compute_divergence = Kernel::<fn(u32, u32)>::new(
        &device,
        &track!(|level, offset| {
            let pos = dispatch_id().xy();
            let target_divergence = charges.read(pos.extend(level)) * 1.0;
            let f = field.read(pos.extend(level));
            let l = -f.x;
            let u = -f.y;
            let r = field.read((pos + Vec2::x()).extend(level)).x;
            let d = field.read((pos + Vec2::y()).extend(level)).y;
            let divergence = r + d + l + u;
            let error = divergence - target_divergence;

            if (pos.x + pos.y) % 2 == offset {
                divergence_error.write(pos.extend(level), error * 1.9);
            } else {
                divergence_error.write(pos.extend(level), 0.0);
            }
        }),
    );
    // +---+---+
    // | u > r |
    // +-^-+-v-+
    // | l < d |
    // +---+---+
    let compute_curl = Kernel::<fn(u32)>::new(
        &device,
        &track!(|level| {
            let pos = dispatch_id().xy();
            let target_curl = magnets.read(pos.extend(level)) * 1.0;
            let f = field.read((pos + 1).extend(level));

            let r = field.read((pos + Vec2::x()).extend(level)).x;
            let d = f.y;
            let l = -f.x;
            let u = -field.read((pos + Vec2::y()).extend(level)).y;

            let curl = r + d + l + u;
            let error = curl - target_curl;

            curl_error.write(pos.extend(level), error);
        }),
    );
    // Note: We don't really gain anything from splitting the divergence and curl calculations.
    // So if we're running both of them per frame then this is preferable.
    let apply_deltas = Kernel::<fn(u32)>::new(
        &device,
        &track!(|level| {
            let pos = dispatch_id().xy();
            let delta = Vec2::splat(0.0_f32).var();
            // Divergence
            *delta -= -divergence_error.read(pos.extend(level));
            if pos.x > 0 {
                *delta.x -= divergence_error.read((pos - Vec2::x()).extend(level));
            }
            if pos.y > 0 {
                *delta.y -= divergence_error.read((pos - Vec2::y()).extend(level));
            }
            // Curl
            if pos.x > 0 && pos.y > 0 {
                *delta -= curl_error.read((pos - 1).extend(level)) * Vec2::new(-1.0, 1.0);
            }
            if pos.x > 0 && (pos.y + 1) < GRID_SIZE << level && pos.x < GRID_SIZE << level {
                *delta.x -= curl_error.read((pos - Vec2::x()).extend(level));
            }
            if pos.y > 0 && (pos.x + 1) < GRID_SIZE << level && pos.y < GRID_SIZE << level {
                *delta.y -= -curl_error.read((pos - Vec2::y()).extend(level));
            }
            field.write(
                pos.extend(level),
                field.read(pos.extend(level)) + delta / 4.0,
            );
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
                let vel = particle_velocity.read(pos) + field.read(pos.extend(0)) * 1.0 / 30.0;
                let movement = vel * dt;
                /* field.read(pos.extend(0));
                if (movement == Vec2::splat(0.0)).any() {
                    particles.write(Vec2::new(0, 0), 1);
                }
                let movement = movement.normalize();  */
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
                    // particle_velocity.write(new_pos, vel);
                    particles.write(pos, 0);
                } else {
                    // particle_velocity.write(pos, vel);
                }
            }
        }),
    );
    let update_trail_kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let pos = dispatch_id().xy();
            let p = particles.read(pos);
            let t = trail.read(pos);
            if p == 1 {
                trail.write(pos, 1.0);
            } else if t > 0.0 {
                trail.write(pos, t - 0.02);
            }
        }),
    );

    let downscale_charges_kernel = Kernel::<fn(u32)>::new(
        &device,
        &track!(|level| {
            let target = dispatch_id().xy();
            let pos = dispatch_id().xy() * 2;
            let c = charges.read(pos.extend(level - 1))
                + charges.read((pos + Vec2::x()).extend(level - 1))
                + charges.read((pos + Vec2::y()).extend(level - 1))
                + charges.read((pos + Vec2::x() + Vec2::y()).extend(level - 1));
            charges.write(target.extend(level), c);
            let m = magnets.read(pos.extend(level - 1))
                + magnets.read((pos + Vec2::x()).extend(level - 1))
                + magnets.read((pos + Vec2::y()).extend(level - 1))
                + magnets.read((pos + Vec2::x() + Vec2::y()).extend(level - 1));
            magnets.write(target.extend(level), m);
        }),
    );

    let upscale_field_kernel = Kernel::<fn(u32)>::new(
        &device,
        &track!(|level| {
            let pos = dispatch_id().xy();
            let read_pos = pos / 2;
            let v = field.read(read_pos.extend(level + 1)).var();
            // let dv = divergence_error.read(read_pos.extend(level + 1)) / 2.0;
            let factor = 1.0; // - (1.0 / (1.0 + dv * dv));
            if pos.x % 2 == 1 {
                *v.x = (v.x + field.read((read_pos + Vec2::x()).extend(level + 1)).x) / 2.0;
            }
            if pos.y % 2 == 1 {
                *v.y = (v.y + field.read((read_pos + Vec2::y()).extend(level + 1)).y) / 2.0;
            }
            let ov = field.read(pos.extend(level));
            field.write(pos.extend(level), v / 2.0 * factor + ov * (1.0 - factor));
        }),
    );

    let write_charge_kernel = Kernel::<fn(Vec2<u32>, f32)>::new(
        &device,
        &track!(|pos, value| {
            charges.write((pos + dispatch_id().xy()).extend(0), value);
        }),
    );
    let write_magnet_kernel = Kernel::<fn(Vec2<u32>, f32)>::new(
        &device,
        &track!(|pos, value| {
            magnets.write((pos + dispatch_id().xy()).extend(0), value);
        }),
    );
    let write_particle_kernel = Kernel::<fn(Vec2<u32>)>::new(
        &device,
        &track!(|pos| {
            particles.write(pos + dispatch_id().xy(), 1);
        }),
    );

    let mut active_buttons = HashSet::new();

    let mut update_cursor = |active_buttons: &HashSet<MouseButton>, rt: &mut Runtime| {
        let pos = Vec2::new(
            (rt.cursor_pos.x as u32) / SCALING,
            (rt.cursor_pos.y as u32) / SCALING,
        );
        if active_buttons.contains(&MouseButton::Left) {
            write_charge_kernel.dispatch([1, 1, 1], &pos, &-CHARGE);
        }
        if active_buttons.contains(&MouseButton::Right) {
            write_magnet_kernel.dispatch([1, 1, 1], &pos, &CHARGE);
        }
        if active_buttons.contains(&MouseButton::Middle) {
            write_particle_kernel.dispatch([1, 1, 1], &pos);
        }
    };
    let update_cursor = &mut update_cursor;

    let mut update_keyboard = |ev: KeyEvent, rt: &mut Runtime| {
        if ev.state != ElementState::Pressed {
            return;
        }
        let PhysicalKey::Code(key) = ev.physical_key else {
            panic!("Invalid")
        };
        match key {
            KeyCode::KeyL => {
                rt.viewed_layer += 1;
                if rt.viewed_layer >= DOWNSCALE_COUNT {
                    rt.viewed_layer = 0;
                }
            }
            KeyCode::KeyK => {
                if rt.viewed_layer == 0 {
                    rt.viewed_layer = DOWNSCALE_COUNT - 1;
                } else {
                    rt.viewed_layer -= 1;
                }
            }
            KeyCode::KeyD => {
                rt.view.next();
            }
            KeyCode::KeyM => {
                rt.activate_multigrid = !rt.activate_multigrid;
            }
            _ => (),
        }
    };
    let update_keyboard = &mut update_keyboard;

    let mut rt = Runtime::default();

    let start = Instant::now();

    let dt = Duration::from_secs_f64(1.0 / 60.0);
    let step = 10.0;

    let mut parity = 0;

    let mut avg_iter_time = 0.0;

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

                    if dt * rt.t < start.elapsed() {
                        let iter_st = Instant::now();
                        rt.t += 1;
                        update_cursor(&active_buttons, &mut rt);
                        {
                            let mut commands = vec![];
                            for i in 1..DOWNSCALE_COUNT {
                                let size = GRID_SIZE >> i;
                                commands.push(
                                    downscale_charges_kernel.dispatch_async([size, size, 1], &i),
                                );
                            }
                            for i in (0..DOWNSCALE_COUNT).rev() {
                                let size = GRID_SIZE >> i;
                                if rt.activate_multigrid && i < DOWNSCALE_COUNT - 1 {
                                    commands.push(
                                        upscale_field_kernel.dispatch_async([size, size, 1], &i),
                                    );
                                }
                                for _ in 0..16 {
                                    commands.extend([
                                        compute_divergence.dispatch_async(
                                            [size, size, 1],
                                            &i,
                                            &parity,
                                        ),
                                        compute_curl.dispatch_async([size - 1, size - 1, 1], &i),
                                        apply_deltas.dispatch_async([size + 1, size + 1, 1], &i),
                                    ]);
                                    parity = 1 - parity;
                                }
                            }
                            commands.extend([
                                update_kernel.dispatch_async(
                                    [GRID_SIZE, GRID_SIZE, 1],
                                    &step,
                                    &rt.t,
                                ),
                                update_trail_kernel.dispatch_async([GRID_SIZE, GRID_SIZE, 1]),
                                draw_kernel.dispatch_async(
                                    [GRID_SIZE * SCALING, GRID_SIZE * SCALING, 1],
                                    &rt.viewed_layer,
                                    &rt.view,
                                ),
                            ]);
                            scope.submit(commands);
                        }
                        avg_iter_time = avg_iter_time * 0.9 + iter_st.elapsed().as_secs_f64() * 0.1;
                        if rt.t % 60 == 0 {
                            println!("Avg iter time: {}", avg_iter_time * 1000.0);
                        }
                    }
                    window.request_redraw();
                }
                WindowEvent::CursorMoved { position, .. } => {
                    rt.cursor_pos = position;
                    update_cursor(&active_buttons, &mut rt);
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
                    update_cursor(&active_buttons, &mut rt);
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    update_keyboard(event, &mut rt);
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
