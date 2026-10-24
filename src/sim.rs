//! Tiled stable-fluids solver advecting a per-cell dye field that relaxes
//! back to the wallpaper's rest state. Only active tiles are simulated; grids
//! are padded with a 1-cell border ring so stencils need no bounds checks.
//! Displacement is clamped to <= 1 cell (CFL <= 1), so upwind bilinear
//! sampling reads only the 3x3 neighbourhood: shifted loads, no gathers.

use fearless_simd::{Level, dispatch, f32x8, prelude::*};
use fearless_simd_macros::simd;

/// Dye tile edge, in cells. Multiple of every native f32 vector width.
pub const TILE: usize = 16;
/// Velocity tile edge, in velocity cells (= SIMD width).
const VTILE: usize = 8;
const L: usize = 8;
const MAX_VEL_CELLS: usize = 96;
/// Jacobi iterations (even: ping-pong ends in the primary buffer).
const PRESSURE_ITERS: usize = 24;
/// Below this speed (dye cells/s) motion is invisible next to relaxation.
const VEL_EPS: f32 = 0.2;
/// Linear drag (dye cells/s^2) so velocity reaches zero in finite time.
const DRAG: f32 = 0.6;
/// Deviations under `DYE_SNAP_RATE * dt` glyph steps snap to rest, ending
/// invisible tails (0.04 at 1/60 s, about one tone step). Scaled by the step
/// so small steps don't erase a larger share of fresh motion.
const DYE_SNAP_RATE: f32 = 2.4;
/// Linear dye relaxation (glyph steps/s) on top of the exponential one.
const DYE_LIN: f32 = 0.35;
/// Substeps per frame cap: frames up to 1/15 s keep steps <= 1/60 s (CFL).
const MAX_SUBSTEPS: f32 = 4.0;
/// Velocity (dye cells/s) added per dye cell of pointer travel. Per distance,
/// not per time, so a gesture injects the same momentum at any frame rate.
const PUSH_GAIN: f32 = 36.0;

#[derive(Clone, Debug)]
pub struct Params {
    /// Velocity damping per second (higher = motion dies faster).
    pub damping: f32,
    /// Dye relaxation time constant toward the rest image, seconds.
    pub relax: f32,
    /// Vorticity confinement strength (swirliness).
    pub vorticity: f32,
    /// Mouse force gain.
    pub force: f32,
    /// Splat radius in screen cells.
    pub radius: f32,
    /// Ambient curl-noise strength.
    pub ambient: f32,
    /// Ambient flow enabled. When off, the sim fully idles once settled.
    pub ambient_on: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            damping: 1.4,
            relax: 1.4,
            vorticity: 3.0,
            force: 1.0,
            radius: 3.5,
            ambient: 0.6,
            ambient_on: false,
        }
    }
}

enum Splat {
    Push { x: f32, y: f32, dx: f32, dy: f32 },
    Burst { x: f32, y: f32, strength: f32 },
    Swirl { x: f32, y: f32, strength: f32 },
}

/// Half-open rect in interior velocity-cell coordinates.
#[derive(Clone, Copy)]
struct Rect {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

pub struct Sim {
    pub p: Params,
    level: Level,
    // Dye: interior dw x dh (multiples of TILE), stride ds = dw + 2.
    dw: usize,
    dh: usize,
    ds: usize,
    pub tiles_x: usize,
    pub tiles_y: usize,
    dye: Vec<f32>,
    dye2: Vec<f32>,
    rest: Vec<f32>,
    dye_on: Vec<bool>,
    adv: Vec<bool>,
    relax_only: Vec<bool>,
    /// Per-tile scratch for one row of tiles.
    tile_max: Vec<f32>,
    /// Dye tiles written since the renderer last looked.
    pub touched: Vec<bool>,
    // Velocity: interior w x h, stride vs = w + 2, zero border ring.
    k: usize,
    w: usize,
    h: usize,
    vs: usize,
    vtx: usize,
    vty: usize,
    u: Vec<f32>,
    v: Vec<f32>,
    u2: Vec<f32>,
    v2: Vec<f32>,
    pr: Vec<f32>,
    pr2: Vec<f32>,
    div: Vec<f32>,
    curl: Vec<f32>,
    vel_on: Vec<bool>,
    // Velocity upsampled along x to dye resolution, as displacement per step:
    // (h + 2) rows of dw.
    ux: Vec<f32>,
    uy: Vec<f32>,
    xi: Vec<u32>,
    xt: Vec<f32>,
    /// Dye row -> (padded velocity row, weight).
    yj: Vec<u32>,
    yt: Vec<f32>,
    splats: Vec<Splat>,
    time: f32,
    vel_active: bool,
    dye_active: bool,
}

/// Every SIMD level this CPU supports, strongest first, with short names.
pub fn simd_levels() -> Vec<(&'static str, Level)> {
    let best = Level::new();
    let mut v = Vec::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Levels compiled out via .cargo/config.toml dispatch one level down.
        if let Some(t) = best.as_avx512()
            && !cfg!(disable_dispatch_avx512)
        {
            v.push(("avx512", Level::Avx512(t)));
        }
        if let Some(t) = best.as_avx2() {
            v.push(("avx2", Level::Avx2(t)));
        }
        if let Some(t) = best.as_sse4_2()
            && !cfg!(disable_dispatch_sse4_2)
        {
            v.push(("sse4.2", Level::Sse4_2(t)));
        }
        if let Some(t) = best.as_sse2() {
            v.push(("sse2", Level::Sse2(t)));
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = best.as_neon() {
        v.push(("neon", Level::Neon(t)));
    }
    if v.is_empty() {
        v.push(("fallback", best));
    }
    v
}

/// `SITU_SIMD=<name>` (see [`simd_levels`]; `off` = weakest) overrides the
/// detected level. Unknown names use the best.
fn level_from_env() -> Level {
    let Some(want) = std::env::var_os("SITU_SIMD") else {
        return Level::new();
    };
    let levels = simd_levels();
    let pick = if want == "off" {
        levels.last()
    } else {
        levels.iter().find(|(n, _)| want == *n)
    };
    pick.map_or_else(Level::new, |&(_, l)| l)
}

impl Sim {
    pub fn new(cols: usize, rows: usize, rest_in: Vec<f32>, p: Params) -> Self {
        let dw = cols.next_multiple_of(TILE);
        let dh = rows.next_multiple_of(TILE);
        let ds = dw + 2;
        let k = cols.max(rows).div_ceil(MAX_VEL_CELLS).max(1);
        let (w, h) = (dw.div_ceil(k).next_multiple_of(VTILE), dh.div_ceil(k));
        let vs = w + 2;
        let (vtx, vty) = (w.div_ceil(VTILE), h.div_ceil(VTILE));

        // Pad by edge replication: the margin behaves like more wallpaper.
        let mut rest = vec![0.0; ds * (dh + 2)];
        for y in 0..dh + 2 {
            let sy = y.saturating_sub(1).min(rows - 1);
            for x in 0..ds {
                rest[y * ds + x] = rest_in[sy * cols + x.saturating_sub(1).min(cols - 1)];
            }
        }
        let (mut xi, mut xt) = (vec![0u32; dw], vec![0f32; dw]);
        for x in 0..dw {
            let vx = (x as f32 + 0.5) / k as f32 - 0.5;
            let f = vx.floor();
            xi[x] = (f as isize + 1).clamp(0, w as isize) as u32;
            xt[x] = vx - f;
        }
        let (mut yj, mut yt) = (vec![0u32; dh], vec![0f32; dh]);
        for y in 0..dh {
            let vy = (y as f32 + 0.5) / k as f32 - 0.5;
            let f = vy.floor();
            yj[y] = (f as isize + 1).clamp(0, h as isize) as u32;
            yt[y] = vy - f;
        }
        let (tiles_x, tiles_y) = (dw / TILE, dh / TILE);
        let nv = vs * (h + 2);
        Sim {
            p,
            level: level_from_env(),
            dw,
            dh,
            ds,
            tiles_x,
            tiles_y,
            dye: rest.clone(),
            dye2: rest.clone(),
            rest,
            dye_on: vec![false; tiles_x * tiles_y],
            adv: vec![false; tiles_x * tiles_y],
            relax_only: vec![false; tiles_x * tiles_y],
            tile_max: vec![0.0; tiles_x],
            touched: vec![true; tiles_x * tiles_y],
            k,
            w,
            h,
            vs,
            vtx,
            vty,
            u: vec![0.0; nv],
            v: vec![0.0; nv],
            u2: vec![0.0; nv],
            v2: vec![0.0; nv],
            pr: vec![0.0; nv],
            pr2: vec![0.0; nv],
            div: vec![0.0; nv],
            curl: vec![0.0; nv],
            vel_on: vec![false; vtx * vty],
            ux: vec![0.0; (h + 2) * dw],
            uy: vec![0.0; (h + 2) * dw],
            xi,
            xt,
            yj,
            yt,
            splats: Vec::new(),
            time: 0.0,
            vel_active: false,
            dye_active: false,
        }
    }

    pub fn level(&self) -> Level {
        self.level
    }

    /// Overrides the runtime-detected SIMD level (tests, A/B benchmarks).
    pub fn with_level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }

    /// Marks every tile touched so a renderer re-quantizes everything.
    pub fn touch_all(&mut self) {
        self.touched.fill(true);
    }

    /// Interior dye row `y` (length `dw` >= `cols`).
    #[inline]
    pub fn dye_row(&self, y: usize) -> &[f32] {
        let o = (y + 1) * self.ds + 1;
        &self.dye[o..o + self.dw]
    }

    fn ambient(&self) -> bool {
        self.p.ambient_on && self.p.ambient > 0.0
    }

    pub fn active(&self) -> bool {
        self.vel_active || self.dye_active || !self.splats.is_empty() || self.ambient()
    }

    pub fn debug(&self) -> (bool, bool, usize, usize) {
        let vt = self.vel_on.iter().filter(|&&b| b).count();
        let dt = self.dye_on.iter().filter(|&&b| b).count();
        (self.vel_active, self.dye_active, vt, dt)
    }

    /// Pointer moved from/to, in dye-cell coordinates.
    pub fn push(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        let (dx, dy) = (x1 - x0, y1 - y0);
        let len = dx.hypot(dy);
        if len < 1e-3 {
            return;
        }
        // Splat along the segment so fast flicks leave a continuous wake.
        let n = (len / self.k as f32).ceil().clamp(1.0, 32.0) as usize;
        for i in 1..=n {
            let t = i as f32 / n as f32;
            self.splats.push(Splat::Push {
                x: x0 + dx * t,
                y: y0 + dy * t,
                dx: dx / n as f32,
                dy: dy / n as f32,
            });
        }
    }

    pub fn burst(&mut self, x: f32, y: f32, strength: f32) {
        self.splats.push(Splat::Burst { x, y, strength });
    }

    pub fn swirl(&mut self, x: f32, y: f32, strength: f32) {
        self.splats.push(Splat::Swirl { x, y, strength });
    }

    /// Advances one app frame of `dt` seconds, in substeps of <= 1/60 s
    /// (for frames up to 1/15 s) so advection stays within CFL <= 1.
    pub fn advance(&mut self, dt: f32) {
        let steps = (dt * 60.0).ceil().clamp(1.0, MAX_SUBSTEPS) as usize;
        for _ in 0..steps {
            self.step(dt / steps as f32);
        }
    }

    /// Advances the simulation by `dt` seconds (must be <= 1/60 for CFL).
    pub fn step(&mut self, dt: f32) {
        let level = self.level;
        dispatch!(level, simd => step_impl(simd, self, dt));
    }

    #[inline]
    fn vtile(&self, i: usize, j: usize) -> usize {
        (j / VTILE) * self.vtx + i / VTILE
    }

    /// Max speed (vel cells/s) allowed: keeps dye displacement <= 1 cell/step.
    fn max_vel(&self) -> f32 {
        60.0 / self.k as f32
    }

    fn apply_splats(&mut self) {
        let k = self.k as f32;
        let r = (self.p.radius / k).max(0.75);
        let r2 = r * r;
        let ext = (r * 2.5).ceil() as isize;
        let maxv = self.max_vel();
        let (w, h, vs) = (self.w as isize, self.h as isize, self.vs);
        for s in std::mem::take(&mut self.splats) {
            let (Splat::Push { x, y, .. } | Splat::Burst { x, y, .. } | Splat::Swirl { x, y, .. }) =
                s;
            let (sx, sy) = (x / k - 0.5, y / k - 0.5);
            let (cx, cy) = (sx.round() as isize, sy.round() as isize);
            for j in (cy - ext).max(0)..(cy + ext + 1).min(h) {
                for i in (cx - ext).max(0)..(cx + ext + 1).min(w) {
                    let (ox, oy) = (i as f32 - sx, j as f32 - sy);
                    let d2 = ox * ox + oy * oy;
                    let g = (-d2 / r2).exp();
                    if g < 0.01 {
                        continue;
                    }
                    let (fx, fy) = match s {
                        // Pointer delta (dye cells) -> velocity (vel cells/s).
                        Splat::Push { dx, dy, .. } => {
                            let gain = self.p.force * PUSH_GAIN / k;
                            (dx * gain, dy * gain)
                        }
                        Splat::Burst { strength, .. } => {
                            let d = d2.sqrt().max(0.5);
                            (ox / d * strength / k, oy / d * strength / k)
                        }
                        Splat::Swirl { strength, .. } => {
                            (-oy * strength / (k * r), ox * strength / (k * r))
                        }
                    };
                    let idx = (j as usize + 1) * vs + i as usize + 1;
                    let (mut nu, mut nv) = (self.u[idx] + fx * g, self.v[idx] + fy * g);
                    let m = nu.hypot(nv);
                    if m > maxv {
                        nu *= maxv / m;
                        nv *= maxv / m;
                    }
                    self.u[idx] = nu;
                    self.v[idx] = nv;
                    let t = self.vtile(i as usize, j as usize);
                    self.vel_on[t] = true;
                }
            }
        }
    }

    fn ambient_force(&mut self, dt: f32) {
        // Cheap analytic curl noise: two drifting sinusoidal stream functions.
        let (a, t) = (self.p.ambient, self.time);
        let s = 6.0 / self.w.max(self.h) as f32;
        for j in 0..self.h {
            let y = j as f32 * s;
            for i in 0..self.w {
                let x = i as f32 * s;
                let dpsi_dy = -(x + 0.13 * t).sin() * (y - 0.11 * t).sin()
                    + 0.85 * (1.7 * y + 0.07 * t).cos() * (2.3 * x).cos();
                let dpsi_dx = (x + 0.13 * t).cos() * (y - 0.11 * t).cos()
                    - 1.15 * (1.7 * y + 0.07 * t).sin() * (2.3 * x).sin();
                let idx = (j + 1) * self.vs + i + 1;
                self.u[idx] += dpsi_dy * a * dt;
                self.v[idx] -= dpsi_dx * a * dt;
            }
        }
        self.vel_on.fill(true);
    }

    /// Bounding rect of active velocity tiles, grown by one tile.
    fn active_rect(&self) -> Option<Rect> {
        let (mut tx0, mut ty0, mut tx1, mut ty1) = (usize::MAX, usize::MAX, 0, 0);
        for ty in 0..self.vty {
            for tx in 0..self.vtx {
                if self.vel_on[ty * self.vtx + tx] {
                    tx0 = tx0.min(tx);
                    ty0 = ty0.min(ty);
                    tx1 = tx1.max(tx + 1);
                    ty1 = ty1.max(ty + 1);
                }
            }
        }
        (tx0 != usize::MAX).then(|| Rect {
            x0: tx0.saturating_sub(1) * VTILE,
            y0: ty0.saturating_sub(1) * VTILE,
            x1: ((tx1 + 1) * VTILE).min(self.w),
            y1: ((ty1 + 1) * VTILE).min(self.h),
        })
    }

    /// Zeroes velocity + pressure in velocity tile `t`.
    fn clear_vtile(&mut self, tx: usize, ty: usize) {
        let (x0, x1) = (tx * VTILE, ((tx + 1) * VTILE).min(self.w));
        for j in ty * VTILE..((ty + 1) * VTILE).min(self.h) {
            let o = (j + 1) * self.vs + 1;
            self.u[o + x0..o + x1].fill(0.0);
            self.v[o + x0..o + x1].fill(0.0);
            self.pr[o + x0..o + x1].fill(0.0);
        }
    }
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Scalar twin of the SIMD upwind sampler (used for row tails).
#[inline(always)]
fn upwind1(s: &[f32], i: usize, st: usize, dx: f32, dy: f32) -> f32 {
    let (tx, ty) = (dx.abs().min(1.0), dy.abs().min(1.0));
    let sx = if dx > 0.0 { i - 1 } else { i + 1 };
    let row = if dy > 0.0 { i - st } else { i + st };
    let diag = if dy > 0.0 { sx - st } else { sx + st };
    let a = lerp(s[i], s[sx], tx);
    let b = lerp(s[row], s[diag], tx);
    lerp(a, b, ty)
}

/// `n` values of `s` from `start`. Slicing every stencil row to the same
/// length up front lets LLVM drop bounds checks in the chunk loops.
#[inline(always)]
fn row(s: &[f32], start: usize, n: usize) -> &[f32] {
    &s[start..start + n]
}

#[inline(always)]
fn chunk<S: Simd>(simd: S, s: &[f32], k: usize) -> f32x8<S> {
    f32x8::from_slice(simd, &s.as_chunks::<L>().0[k])
}

#[inline(always)]
fn put<S: Simd>(v: f32x8<S>, s: &mut [f32], k: usize) {
    v.store_slice(&mut s.as_chunks_mut::<L>().0[k]);
}

/// The 3x3 neighbourhood of a row span of `s`, for upwind sampling.
struct Stencil<'a> {
    c: &'a [f32],
    l: &'a [f32],
    r: &'a [f32],
    u: &'a [f32],
    d: &'a [f32],
    ul: &'a [f32],
    ur: &'a [f32],
    dl: &'a [f32],
    dr: &'a [f32],
}

impl<'a> Stencil<'a> {
    #[inline(always)]
    fn new(s: &'a [f32], i: usize, st: usize, n: usize) -> Self {
        Stencil {
            c: row(s, i, n),
            l: row(s, i - 1, n),
            r: row(s, i + 1, n),
            u: row(s, i - st, n),
            d: row(s, i + st, n),
            ul: row(s, i - st - 1, n),
            ur: row(s, i - st + 1, n),
            dl: row(s, i + st - 1, n),
            dr: row(s, i + st + 1, n),
        }
    }

    /// Upwind bilinear sample of chunk `k` displaced by `(dx, dy)` (<= 1 cell).
    #[inline(always)]
    fn sample<S: Simd>(&self, simd: S, k: usize, dx: f32x8<S>, dy: f32x8<S>) -> f32x8<S> {
        let (zero, one) = (f32x8::splat(simd, 0.0), f32x8::splat(simd, 1.0));
        let ld = |s: &[f32]| chunk(simd, s, k);
        let (mx, my) = (dx.simd_gt(zero), dy.simd_gt(zero));
        let (tx, ty) = (dx.abs().min(one), dy.abs().min(one));
        let c = ld(self.c);
        let side = mx.select(ld(self.l), ld(self.r));
        let row = my.select(ld(self.u), ld(self.d));
        let diag = my.select(
            mx.select(ld(self.ul), ld(self.ur)),
            mx.select(ld(self.dl), ld(self.dr)),
        );
        let a = (side - c).mul_add(tx, c);
        let b = (diag - row).mul_add(tx, row);
        (b - a).mul_add(ty, a)
    }
}

/// `dst[base..base+n] = src` advected by displacement `(du, dv) * scale`
/// (indices into `du/dv` are offset by `doff - base`).
#[simd]
#[allow(clippy::too_many_arguments)]
fn advect<S: Simd>(
    simd: S,
    src: &[f32],
    dst: &mut [f32],
    du: &[f32],
    dv: &[f32],
    doff: usize,
    scale: f32,
    st: usize,
    base: usize,
    n: usize,
) {
    const { assert!(TILE.is_multiple_of(L) && VTILE.is_multiple_of(L)) };
    let sc = f32x8::splat(simd, scale);
    let sten = Stencil::new(src, base, st, n);
    let (du, dv, out) = (row(du, doff, n), row(dv, doff, n), &mut dst[base..base + n]);
    let full = n / L;
    for k in 0..full {
        let (dx, dy) = (chunk(simd, du, k) * sc, chunk(simd, dv, k) * sc);
        put(sten.sample(simd, k, dx, dy), out, k);
    }
    for x in full * L..n {
        out[x] = upwind1(src, base + x, st, du[x] * scale, dv[x] * scale);
    }
}

/// Per-step dye relaxation toward rest: `|d - r| * decay - lin`, then
/// deviations under `snap` are dropped.
#[derive(Clone, Copy)]
struct Relax {
    decay: f32,
    lin: f32,
    snap: f32,
}

/// Relaxes `d` toward `r` in place; returns the max remaining |d - r|.
#[simd]
fn relax<S: Simd>(simd: S, d: &mut [f32], r: &[f32], rx: Relax) -> f32 {
    let Relax {
        decay,
        lin,
        snap: snap1,
    } = rx;
    let (vdecay, vlin) = (f32x8::splat(simd, decay), f32x8::splat(simd, lin));
    let (zero, snap) = (f32x8::splat(simd, 0.0), f32x8::splat(simd, snap1));
    let r = &r[..d.len()];
    let (dc, dt) = d.as_chunks_mut::<L>();
    let (rc, rt) = r.as_chunks::<L>();
    let mut acc = zero;
    for (dc, rc) in dc.iter_mut().zip(rc) {
        let (dv, rv) = (f32x8::from_slice(simd, dc), f32x8::from_slice(simd, rc));
        let diff = dv - rv;
        let mag = diff.abs().mul_sub(vdecay, vlin).max(zero);
        let mag = mag.simd_lt(snap).select(zero, mag);
        acc = acc.max(mag);
        (rv + mag.copysign(diff)).store_slice(dc);
    }
    let mut m = acc.reduce_max();
    for (d, &r) in dt.iter_mut().zip(rt) {
        let diff = *d - r;
        let mut mag = (diff.abs() * decay - lin).max(0.0);
        if mag < snap1 {
            mag = 0.0;
        }
        m = m.max(mag);
        *d = r + mag.copysign(diff);
    }
    m
}

#[simd]
fn curl_row<S: Simd>(
    simd: S,
    u: &[f32],
    v: &[f32],
    curl: &mut [f32],
    st: usize,
    base: usize,
    n: usize,
) {
    let h = f32x8::splat(simd, 0.5);
    let (vl, vr) = (row(v, base - 1, n), row(v, base + 1, n));
    let (uu, ud) = (row(u, base - st, n), row(u, base + st, n));
    let out = &mut curl[base..base + n];
    let full = n / L;
    for k in 0..full {
        let ld = |s: &[f32]| chunk(simd, s, k);
        put(((ld(vr) - ld(vl)) - (ld(ud) - ld(uu))) * h, out, k);
    }
    for x in full * L..n {
        out[x] = 0.5 * ((vr[x] - vl[x]) - (ud[x] - uu[x]));
    }
}

/// Vorticity confinement force over a row span.
#[simd]
#[allow(clippy::too_many_arguments)]
fn confine_row<S: Simd>(
    simd: S,
    u: &mut [f32],
    v: &mut [f32],
    curl: &[f32],
    eps: f32,
    st: usize,
    base: usize,
    n: usize,
) {
    let (h, e, tiny) = (
        f32x8::splat(simd, 0.5),
        f32x8::splat(simd, eps),
        f32x8::splat(simd, 1e-5),
    );
    let (c, l, r) = (
        row(curl, base, n),
        row(curl, base - 1, n),
        row(curl, base + 1, n),
    );
    let (cu, cd) = (row(curl, base - st, n), row(curl, base + st, n));
    let (u, v) = (&mut u[base..base + n], &mut v[base..base + n]);
    let full = n / L;
    for k in 0..full {
        let ld = |s: &[f32]| chunk(simd, s, k);
        let nx = (ld(r).abs() - ld(l).abs()) * h;
        let ny = (ld(cd).abs() - ld(cu).abs()) * h;
        let s = e * ld(c) / (nx.mul_add(nx, ny * ny).sqrt() + tiny);
        put(ny.mul_add(s, chunk(simd, u, k)), u, k);
        put(chunk(simd, v, k) - nx * s, v, k);
    }
    for x in full * L..n {
        let nx = 0.5 * (r[x].abs() - l[x].abs());
        let ny = 0.5 * (cd[x].abs() - cu[x].abs());
        let s = eps * c[x] / ((nx * nx + ny * ny).sqrt() + 1e-5);
        u[x] += ny * s;
        v[x] -= nx * s;
    }
}

/// One Jacobi sweep of the pressure Poisson equation over a row segment.
#[simd]
fn jacobi<S: Simd>(
    simd: S,
    p: &[f32],
    out: &mut [f32],
    div: &[f32],
    st: usize,
    base: usize,
    n: usize,
) {
    let q = f32x8::splat(simd, 0.25);
    let (l, r) = (row(p, base - 1, n), row(p, base + 1, n));
    let (u, d) = (row(p, base - st, n), row(p, base + st, n));
    let (dv, out) = (row(div, base, n), &mut out[base..base + n]);
    let full = n / L;
    for k in 0..full {
        let ld = |s: &[f32]| chunk(simd, s, k);
        put((ld(l) + ld(r) + ld(u) + ld(d) - ld(dv)) * q, out, k);
    }
    for x in full * L..n {
        out[x] = (l[x] + r[x] + u[x] + d[x] - dv[x]) * 0.25;
    }
}

#[simd]
fn divergence<S: Simd>(
    simd: S,
    u: &[f32],
    v: &[f32],
    div: &mut [f32],
    st: usize,
    base: usize,
    n: usize,
) {
    let h = f32x8::splat(simd, 0.5);
    let (ul, ur) = (row(u, base - 1, n), row(u, base + 1, n));
    let (vu, vd) = (row(v, base - st, n), row(v, base + st, n));
    let out = &mut div[base..base + n];
    let full = n / L;
    for k in 0..full {
        let ld = |s: &[f32]| chunk(simd, s, k);
        put((ld(ur) - ld(ul) + ld(vd) - ld(vu)) * h, out, k);
    }
    for x in full * L..n {
        out[x] = 0.5 * (ur[x] - ul[x] + vd[x] - vu[x]);
    }
}

#[simd]
#[allow(clippy::too_many_arguments)]
fn subtract_gradient<S: Simd>(
    simd: S,
    u: &mut [f32],
    v: &mut [f32],
    p: &[f32],
    st: usize,
    base: usize,
    n: usize,
) {
    let h = f32x8::splat(simd, 0.5);
    let (l, r) = (row(p, base - 1, n), row(p, base + 1, n));
    let (pu, pd) = (row(p, base - st, n), row(p, base + st, n));
    let (u, v) = (&mut u[base..base + n], &mut v[base..base + n]);
    let full = n / L;
    for k in 0..full {
        let ld = |s: &[f32]| chunk(simd, s, k);
        put(chunk(simd, u, k) - (ld(r) - ld(l)) * h, u, k);
        put(chunk(simd, v, k) - (ld(pd) - ld(pu)) * h, v, k);
    }
    for x in full * L..n {
        u[x] -= 0.5 * (r[x] - l[x]);
        v[x] -= 0.5 * (pd[x] - pu[x]);
    }
}

/// Exponential damping + linear drag + speed clamp; returns max speed.
#[simd]
fn damp<S: Simd>(simd: S, u: &mut [f32], v: &mut [f32], damp: f32, drag: f32, maxv: f32) -> f32 {
    let zero = f32x8::splat(simd, 0.0);
    let (vd, vg) = (f32x8::splat(simd, damp), f32x8::splat(simd, drag));
    let (vm, tiny) = (f32x8::splat(simd, maxv), f32x8::splat(simd, 1e-12));
    let v = &mut v[..u.len()];
    let (uc, ut) = u.as_chunks_mut::<L>();
    let (vc, vt) = v.as_chunks_mut::<L>();
    let mut acc = zero;
    for (uc, vc) in uc.iter_mut().zip(vc) {
        let (a, b) = (f32x8::from_slice(simd, uc), f32x8::from_slice(simd, vc));
        let m = a.mul_add(a, b * b).sqrt();
        let nm = m.mul_sub(vd, vg).max(zero).min(vm);
        let f = nm / m.max(tiny);
        acc = acc.max(nm);
        (a * f).store_slice(uc);
        (b * f).store_slice(vc);
    }
    let mut mx = acc.reduce_max();
    for (u, v) in ut.iter_mut().zip(vt) {
        let m = (*u * *u + *v * *v).sqrt();
        let nm = (m * damp - drag).max(0.0).min(maxv);
        let f = nm / m.max(1e-12);
        *u *= f;
        *v *= f;
        mx = mx.max(nm);
    }
    mx
}

/// Fused dye update over a row span: displacement = lerp of two upsampled
/// velocity rows, upwind advection `src -> dst`, then relaxation toward
/// `rest`. Folds max |dst - rest| into `tile_max` (one entry per TILE).
#[simd]
#[allow(clippy::too_many_arguments)]
fn dye_span<S: Simd>(
    simd: S,
    src: &[f32],
    dst: &mut [f32],
    rest: &[f32],
    (ua, ub, va, vb): (&[f32], &[f32], &[f32], &[f32]),
    fy: f32,
    Relax { decay, lin, snap }: Relax,
    st: usize,
    base: usize,
    tile_max: &mut [f32],
) {
    const { assert!(TILE.is_multiple_of(L)) };
    let zero = f32x8::splat(simd, 0.0);
    let (vfy, vdecay, vlin, snap) = (
        f32x8::splat(simd, fy),
        f32x8::splat(simd, decay),
        f32x8::splat(simd, lin),
        f32x8::splat(simd, snap),
    );
    let n = tile_max.len() * TILE;
    let sten = Stencil::new(src, base, st, n);
    let (ua, ub, va, vb) = (row(ua, 0, n), row(ub, 0, n), row(va, 0, n), row(vb, 0, n));
    let (rest, out) = (row(rest, base, n), &mut dst[base..base + n]);
    for (ti, tm) in tile_max.iter_mut().enumerate() {
        let mut acc = zero;
        for k in ti * (TILE / L)..(ti + 1) * (TILE / L) {
            let ld = |s: &[f32]| chunk(simd, s, k);
            let (a, b) = (ld(ua), ld(va));
            let dx = (ld(ub) - a).mul_add(vfy, a);
            let dy = (ld(vb) - b).mul_add(vfy, b);
            let d = sten.sample(simd, k, dx, dy);
            let r = ld(rest);
            let diff = d - r;
            let mag = diff.abs().mul_sub(vdecay, vlin).max(zero);
            let mag = mag.simd_lt(snap).select(zero, mag);
            acc = acc.max(mag);
            put(r + mag.copysign(diff), out, k);
        }
        *tm = tm.max(acc.reduce_max());
    }
}

/// Runs of consecutive set flags: (start, len).
fn runs(flags: &[bool]) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut x = 0;
    std::iter::from_fn(move || {
        while x < flags.len() && !flags[x] {
            x += 1;
        }
        let s = x;
        while x < flags.len() && flags[x] {
            x += 1;
        }
        (x > s).then_some((s, x - s))
    })
}

#[simd]
fn step_impl<S: Simd>(simd: S, s: &mut Sim, dt: f32) {
    s.time += dt;
    if !s.splats.is_empty() {
        s.apply_splats();
    }
    if s.ambient() {
        s.ambient_force(dt);
    }

    let rect = s.active_rect();
    if let Some(r) = rect {
        velocity_step(simd, s, r, dt);
    }
    s.vel_active = s.vel_on.iter().any(|&b| b);
    dye_step(simd, s, rect, dt);
}

#[simd]
fn velocity_step<S: Simd>(simd: S, s: &mut Sim, r: Rect, dt: f32) {
    let (vs, n) = (s.vs, r.x1 - r.x0);
    let row = |j: usize| (j + 1) * vs + r.x0 + 1;

    // Vorticity confinement. Curl on the rect grown by one vel tile in x
    // (keeps spans 8-aligned) and one row in y; border cells stay zero.
    let (cx0, cx1) = (r.x0.saturating_sub(VTILE), (r.x1 + VTILE).min(s.w));
    let (cy0, cy1) = (r.y0.saturating_sub(1), (r.y1 + 1).min(s.h));
    for j in cy0..cy1 {
        curl_row(
            simd,
            &s.u,
            &s.v,
            &mut s.curl,
            vs,
            (j + 1) * vs + cx0 + 1,
            cx1 - cx0,
        );
    }
    let eps = s.p.vorticity * dt;
    for j in r.y0..r.y1 {
        confine_row(simd, &mut s.u, &mut s.v, &s.curl, eps, vs, row(j), n);
    }
    // No stale curl for the next (possibly smaller) rect.
    for j in cy0..cy1 {
        let o = (j + 1) * vs + 1;
        s.curl[o + cx0..o + cx1].fill(0.0);
    }

    // Self-advection (ping-pong through u2/v2).
    for j in r.y0..r.y1 {
        let b = row(j);
        advect(simd, &s.u, &mut s.u2, &s.u, &s.v, b, dt, vs, b, n);
        advect(simd, &s.v, &mut s.v2, &s.u, &s.v, b, dt, vs, b, n);
    }
    for j in r.y0..r.y1 {
        let b = row(j);
        s.u[b..b + n].copy_from_slice(&s.u2[b..b + n]);
        s.v[b..b + n].copy_from_slice(&s.v2[b..b + n]);
    }

    // Projection. Pressure outside the rect is a fixed (Dirichlet) boundary;
    // seed the ping-pong buffer's halo so both buffers agree on it.
    for j in r.y0..r.y1 {
        divergence(simd, &s.u, &s.v, &mut s.div, vs, row(j), n);
    }
    let (hy0, hy1) = (r.y0, r.y1 + 2);
    let (hx0, hx1) = (r.x0, r.x1 + 2);
    for j in hy0..hy1 {
        let o = j * vs;
        s.pr2[o + hx0..o + hx1].copy_from_slice(&s.pr[o + hx0..o + hx1]);
    }
    for _ in 0..PRESSURE_ITERS / 2 {
        for j in r.y0..r.y1 {
            jacobi(simd, &s.pr, &mut s.pr2, &s.div, vs, row(j), n);
        }
        for j in r.y0..r.y1 {
            jacobi(simd, &s.pr2, &mut s.pr, &s.div, vs, row(j), n);
        }
    }
    for j in r.y0..r.y1 {
        subtract_gradient(simd, &mut s.u, &mut s.v, &s.pr, vs, row(j), n);
    }

    // Damping, per tile, switching settled tiles off (and exactly to zero).
    let damp_f = (-dt * s.p.damping).exp();
    let drag = DRAG * dt / s.k as f32;
    let maxv = s.max_vel();
    for ty in r.y0 / VTILE..r.y1.div_ceil(VTILE) {
        for tx in r.x0 / VTILE..r.x1.div_ceil(VTILE) {
            let (x0, x1) = (tx * VTILE, ((tx + 1) * VTILE).min(s.w));
            let mut m = 0f32;
            for j in ty * VTILE..((ty + 1) * VTILE).min(s.h) {
                let o = (j + 1) * vs + 1;
                m = m.max(damp(
                    simd,
                    &mut s.u[o + x0..o + x1],
                    &mut s.v[o + x0..o + x1],
                    damp_f,
                    drag,
                    maxv,
                ));
            }
            let on = m * s.k as f32 >= VEL_EPS;
            s.vel_on[ty * s.vtx + tx] = on;
            if !on {
                s.clear_vtile(tx, ty);
            }
        }
    }
}

#[simd]
fn dye_step<S: Simd>(simd: S, s: &mut Sim, rect: Option<Rect>, dt: f32) {
    let (ds, dw, k) = (s.ds, s.dw, s.k);
    let rx = Relax {
        decay: (-dt / s.p.relax).exp(),
        lin: DYE_LIN * dt,
        snap: DYE_SNAP_RATE * dt,
    };

    // Which dye tiles see velocity this step (bilinear support: +-1 vel cell).
    s.adv.fill(false);
    let mut any_adv = false;
    let (mut ax0, mut ax1, mut ay0, mut ay1) = (usize::MAX, 0, usize::MAX, 0);
    if let Some(r) = rect {
        let (dx0, dy0) = (r.x0.saturating_sub(1) * k, r.y0.saturating_sub(1) * k);
        let (dx1, dy1) = (((r.x1 + 1) * k).min(dw), ((r.y1 + 1) * k).min(s.dh));
        for ty in dy0 / TILE..dy1.div_ceil(TILE) {
            for tx in dx0 / TILE..dx1.div_ceil(TILE) {
                s.adv[ty * s.tiles_x + tx] = true;
            }
        }
        (ax0, ax1, ay0, ay1) = (
            dx0 / TILE * TILE,
            dx1.div_ceil(TILE) * TILE,
            dy0 / TILE * TILE,
            dy1.div_ceil(TILE) * TILE,
        );
        any_adv = ax1 > ax0;
    }

    if any_adv {
        // Upsample velocity rows along x (as per-step dye displacement).
        let scale = k as f32 * dt;
        let (j0, j1) = (s.yj[ay0] as usize, s.yj[ay1 - 1] as usize + 1);
        for j in j0..=j1 {
            let (ur, vr) = (
                &s.u[j * s.vs..(j + 1) * s.vs],
                &s.v[j * s.vs..(j + 1) * s.vs],
            );
            let (ux, uy) = (
                &mut s.ux[j * dw + ax0..j * dw + ax1],
                &mut s.uy[j * dw + ax0..j * dw + ax1],
            );
            for (((ux, uy), &i), &t) in ux
                .iter_mut()
                .zip(uy)
                .zip(&s.xi[ax0..ax1])
                .zip(&s.xt[ax0..ax1])
            {
                let i = i as usize;
                *ux = lerp(ur[i], ur[i + 1], t) * scale;
                *uy = lerp(vr[i], vr[i + 1], t) * scale;
            }
        }

        for ty in 0..s.tiles_y {
            let flags = &s.adv[ty * s.tiles_x..(ty + 1) * s.tiles_x];
            for (tx0, nt) in runs(flags) {
                let (x0, xn) = (tx0 * TILE, nt * TILE);
                let tm = &mut s.tile_max[..nt];
                tm.fill(0.0);
                for y in ty * TILE..(ty + 1) * TILE {
                    let (j, fy) = (s.yj[y] as usize, s.yt[y]);
                    let (a, b) = (j * dw + x0, (j + 1) * dw + x0);
                    let rows = (
                        &s.ux[a..a + xn],
                        &s.ux[b..b + xn],
                        &s.uy[a..a + xn],
                        &s.uy[b..b + xn],
                    );
                    let base = (y + 1) * ds + x0 + 1;
                    dye_span(
                        simd,
                        &s.dye,
                        &mut s.dye2,
                        &s.rest,
                        rows,
                        fy,
                        rx,
                        ds,
                        base,
                        tm,
                    );
                }
                for (i, &m) in tm.iter().enumerate() {
                    let t = ty * s.tiles_x + tx0 + i;
                    s.dye_on[t] = m > 0.0;
                    s.touched[t] = true;
                }
                // Publish (advection read the old field for the whole step).
                for y in ty * TILE..(ty + 1) * TILE {
                    let b = (y + 1) * ds + x0 + 1;
                    s.dye[b..b + xn].copy_from_slice(&s.dye2[b..b + xn]);
                }
            }
        }
    }

    // Tiles that only need to relax back home (in place), in runs.
    for ((r, &on), &adv) in s.relax_only.iter_mut().zip(&s.dye_on).zip(&s.adv) {
        *r = on && !adv;
    }
    let mut any_relax = false;
    for ty in 0..s.tiles_y {
        let flags = &s.relax_only[ty * s.tiles_x..(ty + 1) * s.tiles_x];
        for (tx0, nt) in runs(flags) {
            let (x0, xn) = (tx0 * TILE, nt * TILE);
            let tm = &mut s.tile_max[..nt];
            tm.fill(0.0);
            for y in ty * TILE..(ty + 1) * TILE {
                let b = (y + 1) * ds + x0 + 1;
                let (d, r) = (
                    s.dye[b..b + xn].as_chunks_mut::<TILE>().0,
                    s.rest[b..b + xn].as_chunks::<TILE>().0,
                );
                for ((d, r), m) in d.iter_mut().zip(r).zip(tm.iter_mut()) {
                    *m = m.max(relax(simd, d, r, rx));
                }
            }
            for (i, &m) in tm.iter().enumerate() {
                let t = ty * s.tiles_x + tx0 + i;
                s.dye_on[t] = m > 0.0;
                s.touched[t] = true;
            }
            any_relax = true;
        }
    }
    s.dye_active = s.dye_on.iter().any(|&b| b);

    if any_adv || any_relax {
        // Clamp-to-edge border ring for the next step's stencils.
        let last = (s.dh + 1) * ds;
        s.dye.copy_within(ds..2 * ds, 0);
        s.dye.copy_within(last - ds..last, last);
        for y in 0..s.dh + 2 {
            let o = y * ds;
            s.dye[o] = s.dye[o + 1];
            s.dye[o + dw + 1] = s.dye[o + dw];
        }
    }
}
