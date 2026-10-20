//! Scripted interaction shared by tests, benchmarks and the preview tool:
//! a pointer swipe, a click burst and a scroll swirl, then settling.

use crate::sim::Sim;

/// Seconds the scripted pointer moves: out along the path, then back.
pub const SWIPE: f32 = 1.6;
/// Episodes that have not settled after this many seconds are cut off.
pub const MAX_SECONDS: f32 = 20.0;

/// Scripted pointer position (in cells) at time `t`. It retraces its path
/// and ends where it started, so recorded episodes loop seamlessly.
pub fn pointer(t: f32, cols: usize, rows: usize) -> (f32, f32) {
    let half = SWIPE / 2.0;
    let t = t.clamp(0.0, SWIPE);
    let u = if t <= half { t } else { SWIPE - t };
    let (c, r) = (cols as f32, rows as f32);
    (
        c * (0.2 + 0.6 * u / half),
        r * (0.5 + 0.2 * (u * 7.0).sin()),
    )
}

/// Applies the scripted input for the frame `(t - dt, t]`.
pub fn input(sim: &mut Sim, t: f32, dt: f32, cols: usize, rows: usize) {
    let t0 = t - dt;
    if t0 < SWIPE {
        let (a, b) = (pointer(t0, cols, rows), pointer(t, cols, rows));
        sim.push(a.0, a.1, b.0, b.1);
    }
    let (c, r) = (cols as f32, rows as f32);
    if (t0..t).contains(&0.3) {
        sim.burst(c * 0.5, r * 0.5, 40.0);
    }
    if (t0..t).contains(&0.5) {
        sim.swirl(c * 0.3, r * 0.6, 25.0);
    }
}

/// Runs one episode at `fps`: scripted input, then until the sim settles.
/// Calls `frame(sim, t)` after each frame's simulation steps (render there).
/// Returns the number of frames.
pub fn episode(
    sim: &mut Sim,
    fps: f32,
    cols: usize,
    rows: usize,
    mut frame: impl FnMut(&mut Sim, f32),
) -> usize {
    let dt = 1.0 / fps;
    // Same substepping as the app: steps <= 1/60 s keep advection CFL <= 1.
    let steps = (dt * 60.0).ceil().max(1.0) as usize;
    let mut f = 0;
    loop {
        f += 1;
        let t = f as f32 * dt;
        input(sim, t, dt, cols, rows);
        for _ in 0..steps {
            sim.step(dt / steps as f32);
        }
        frame(sim, t);
        if (t > SWIPE && !sim.active()) || t > MAX_SECONDS {
            return f;
        }
    }
}
