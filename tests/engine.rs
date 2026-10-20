//! End-to-end engine invariants (no compositor): the sim settles back to
//! the exact rest image, incremental multi-buffer painting matches a full
//! repaint, damage covers every changed pixel, and every SIMD level agrees.

use fearless_simd::Level;
use situ::Config;
use situ::render::{Layout, Renderer};
use situ::sim::{Sim, simd_levels};
use situ::workload;

const W: usize = 640;
const H: usize = 400;

fn config() -> Config {
    let mut c = Config::default();
    c.style.rows = 25.0;
    c.fps = 60.0;
    c
}

fn layout(cfg: &Config) -> Layout {
    Layout::new(W, H, cfg.style.cell_px(H))
}

fn setup(cfg: &Config, level: Level) -> (Sim, Renderer) {
    let l = layout(cfg);
    let rest = cfg.style.generate(l.cols, l.rows);
    let sim = Sim::new(l.cols, l.rows, rest, cfg.params.clone()).with_level(level);
    (sim, Renderer::new(l, &cfg.style))
}

fn bytes(px: &mut [u32]) -> &mut [u8] {
    // SAFETY: plain-old-data view of an aligned u32 buffer.
    unsafe { std::slice::from_raw_parts_mut(px.as_mut_ptr().cast(), px.len() * 4) }
}

/// Full repaint of the sim's current state with a fresh renderer.
fn reference(cfg: &Config, sim: &mut Sim) -> Vec<u32> {
    let mut r = Renderer::new(layout(cfg), &cfg.style);
    let s = r.add_buffer();
    let mut px = vec![0u32; W * H];
    sim.touch_all();
    r.sync(sim);
    r.paint(sim.level(), s, bytes(&mut px), W * 4);
    px
}

#[test]
fn settles_to_exact_rest_image() {
    let cfg = config();
    let (mut sim, mut r) = setup(&cfg, Level::new());
    let (cols, rows) = (r.layout.cols, r.layout.rows);
    let s = r.add_buffer();
    let mut px = vec![0u32; W * H];
    r.sync(&mut sim);
    r.paint(sim.level(), s, bytes(&mut px), W * 4);
    let rest = px.clone();

    let mut stirred = false;
    workload::episode(&mut sim, cfg.fps, cols, rows, |sim, _| {
        r.sync(sim);
        r.paint(sim.level(), s, bytes(&mut px), W * 4);
        stirred |= px != rest;
    });
    assert!(stirred, "the workload should change the image");
    assert!(
        !sim.active(),
        "did not settle within {}s",
        workload::MAX_SECONDS
    );
    assert!(px == rest, "settled image differs from the rest image");
}

#[test]
fn incremental_paint_matches_full_repaint_and_damage_covers_changes() {
    let cfg = config();
    let (mut sim, mut r) = setup(&cfg, Level::new());
    let (cols, rows) = (r.layout.cols, r.layout.rows);
    // Three buffers used round-robin, as when the compositor holds some.
    let mut bufs: Vec<(usize, Vec<u32>)> = (0..3)
        .map(|_| (r.add_buffer(), vec![0u32; W * H]))
        .collect();
    let mut shown: Option<Vec<u32>> = None;
    let mut f = 0;
    workload::episode(&mut sim, cfg.fps, cols, rows, |sim, _| {
        r.sync(sim);
        let (s, px) = &mut bufs[f % 3];
        r.paint(sim.level(), *s, bytes(px), W * 4);
        let damage = r.take_damage();
        assert!(
            damage.len() <= 256,
            "frame {f}: {} damage rects",
            damage.len()
        );
        if let Some(prev) = &shown {
            for (i, (a, b)) in px.iter().zip(prev).enumerate() {
                if a != b {
                    let (x, y) = ((i % W) as i32, (i / W) as i32);
                    let covered = damage
                        .iter()
                        .any(|&(dx, dy, dw, dh)| x >= dx && x < dx + dw && y >= dy && y < dy + dh);
                    assert!(covered, "frame {f}: pixel ({x},{y}) changed outside damage");
                }
            }
        }
        shown = Some(px.clone());
        if f % 10 == 0 {
            assert!(
                reference(&cfg, sim) == *px,
                "frame {f}: incremental != full repaint"
            );
        }
        f += 1;
    });
}

/// Dye field after a fixed number of workload frames at `level`.
fn dye_after(cfg: &Config, level: Level, frames: usize) -> Vec<f32> {
    let (mut sim, r) = setup(cfg, level);
    let (cols, rows) = (r.layout.cols, r.layout.rows);
    let dt = 1.0 / cfg.fps;
    for f in 1..=frames {
        workload::input(&mut sim, f as f32 * dt, dt, cols, rows);
        sim.step(dt);
    }
    (0..rows)
        .flat_map(|y| sim.dye_row(y)[..cols].to_vec())
        .collect()
}

#[test]
fn deterministic() {
    let cfg = config();
    assert!(dye_after(&cfg, Level::new(), 40) == dye_after(&cfg, Level::new(), 40));
}

/// Every SIMD path this CPU can run must agree with the weakest one, within
/// FMA rounding differences.
#[test]
fn all_simd_levels_agree() {
    let cfg = config();
    let levels = simd_levels();
    let (base_name, base) = *levels.last().unwrap();
    let reference = dye_after(&cfg, base, 40);
    for &(name, level) in &levels {
        let dye = dye_after(&cfg, level, 40);
        let max = dye
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max < 1e-4,
            "{name} vs {base_name}: max dye difference {max}"
        );
        eprintln!("{name:>7} vs {base_name}: max dye difference {max:e}");
    }
}
