//! Tile-granular software renderer: every (glyph, tone) is pre-rendered into a
//! sprite atlas, so drawing a cell is row copies. Only touched tiles are
//! re-quantized, each shm buffer tracks its own stale tiles, and damage is
//! tight boxes per 4x4-cell block, merged 2x2 when over a cap.

use fearless_simd::{dispatch, f32x8, prelude::*, u32x8};
use fearless_simd_macros::simd;

use crate::field::Style;
use crate::glyph::{mask, shape_at};
use crate::sim::{Sim, TILE};

/// Steps per glyph (tone + morph frames); sim changes below this
/// resolution produce no pixel writes and no damage.
pub const LEVELS: usize = 32;
/// Keys are small integers stored as `f32` so diffing is a SIMD compare.
const UNKNOWN: f32 = -1.0;
/// Damage block edge, in cells (half a SIMD chunk).
const DBLK: usize = 4;
/// More damage rects than this per frame get coarsened.
const MAX_RECTS: usize = 256;

pub struct Layout {
    pub width: usize,
    pub height: usize,
    pub cell: usize,
    pub cols: usize,
    pub rows: usize,
    pub ox: isize,
    pub oy: isize,
}

impl Layout {
    /// Centered cell grid covering a `width` x `height` pixel buffer.
    pub fn new(width: usize, height: usize, cell: usize) -> Self {
        let cell = cell.max(4);
        let cols = width.div_ceil(cell);
        let rows = height.div_ceil(cell);
        let ox = -(((cols * cell) - width) as isize / 2);
        let oy = -(((rows * cell) - height) as isize / 2);
        Layout {
            width,
            height,
            cell,
            cols,
            rows,
            ox,
            oy,
        }
    }

    /// One output's window onto a desktop-wide grid of `lcell`-logical-pixel
    /// cells anchored at the logical origin (`--span`). `pos` is the output's
    /// logical position and `scale` its buffer pixels per logical pixel.
    /// Returns the layout and the global index of its first cell.
    pub fn spanned(
        width: usize,
        height: usize,
        scale: f64,
        lcell: f64,
        pos: (i32, i32),
    ) -> (Self, (i32, i32)) {
        let cell = ((lcell * scale).round() as usize).max(4);
        // This output's logical cell after rounding to whole pixels.
        let lc = cell as f64 / scale;
        let axis = |p: i32, len: usize| {
            let g = (f64::from(p) / lc).floor();
            let off =
                (-((f64::from(p) - g * lc) * scale).round() as isize).clamp(1 - cell as isize, 0);
            (
                g as i32,
                off,
                ((len as isize - off) as usize).div_ceil(cell),
            )
        };
        let (gx, ox, cols) = axis(pos.0, width);
        let (gy, oy, rows) = axis(pos.1, height);
        let layout = Layout {
            width,
            height,
            cell,
            cols,
            rows,
            ox,
            oy,
        };
        (layout, (gx, gy))
    }

    /// Visible pixel rect of cell (i, j), clipped to the buffer.
    #[inline]
    fn cell_rect(&self, i: usize, j: usize) -> (usize, usize, usize, usize) {
        let c = self.cell as isize;
        let clip = |v: isize, max: usize| v.clamp(0, max as isize) as usize;
        (
            clip(self.ox + i as isize * c, self.width),
            clip(self.oy + j as isize * c, self.height),
            clip(self.ox + (i as isize + 1) * c, self.width),
            clip(self.oy + (j as isize + 1) * c, self.height),
        )
    }

    /// Pixel position -> fractional cell coordinates.
    pub fn to_cell(&self, x: f64, y: f64) -> (f32, f32) {
        (
            ((x - self.ox as f64) / self.cell as f64) as f32,
            ((y - self.oy as f64) / self.cell as f64) as f32,
        )
    }
}

/// Per-shm-buffer bookkeeping.
struct BufState {
    shown: Vec<f32>,
    dirty: Vec<bool>,
}

/// Empty damage box sentinel.
const NO_BOX: [u16; 4] = [u16::MAX, u16::MAX, 0, 0];

pub struct Renderer {
    pub layout: Layout,
    /// `nkeys` sprites of `cell * cell` XRGB pixels, key = floor(v * LEVELS).
    atlas: Vec<u32>,
    max_key: f32,
    /// Key -> first key with a pixel-identical sprite, so invisible changes
    /// (blank band, tone steps that round to the same color) cost nothing.
    canon: Vec<f32>,
    /// Current key per cell, row stride `kw` (tile-padded).
    keys: Vec<f32>,
    kw: usize,
    tiles_x: usize,
    tiles_y: usize,
    states: Vec<BufState>,
    /// Pending damage per DBLK x DBLK block, in cells: [x0, y0, x1, y1).
    damage: Vec<[u16; 4]>,
    /// Damage blocks per row.
    dbw: usize,
    /// Indices of non-empty `damage` blocks, so taking damage is O(changes).
    damaged: Vec<u32>,
    /// Scratch for coarsening damage.
    merge_slot: Vec<u32>,
    any_damage: bool,
}

impl Renderer {
    pub fn new(layout: Layout, style: &Style) -> Self {
        let glyphs = style.glyphs();
        let cs = layout.cell;
        let nkeys = glyphs * LEVELS;
        let bg = style.palette.bg.map(f32::from);
        let pack = |c: [f32; 3]| {
            let c = c.map(|v| v.round().clamp(0.0, 255.0) as u32);
            0xFF00_0000 | c[0] << 16 | c[1] << 8 | c[2]
        };
        let mut atlas = vec![pack(bg); nkeys * cs * cs];
        for key in 0..nkeys {
            let v = (key as f32 + 0.5) / LEVELS as f32;
            let (g, t) = shape_at(v, glyphs);
            let fg = if g == 0 {
                // Blank -> first glyph: grow out of the background.
                let c = style.color(1, 0.0).map(f32::from);
                std::array::from_fn(|k| bg[k] + (c[k] - bg[k]) * t)
            } else {
                style.color(g, v - g as f32).map(f32::from)
            };
            let m = mask(&style.ramp, g, t, cs);
            let sprite = &mut atlas[key * cs * cs..(key + 1) * cs * cs];
            for (p, &a) in sprite.iter_mut().zip(&m) {
                let a = f32::from(a) / 255.0;
                *p = pack(std::array::from_fn(|k| bg[k] + (fg[k] - bg[k]) * a));
            }
        }
        let sprite = |k: usize| &atlas[k * cs * cs..(k + 1) * cs * cs];
        let mut canon = vec![0.0; nkeys];
        for k in 1..nkeys {
            canon[k] = if sprite(k) == sprite(k - 1) {
                canon[k - 1]
            } else {
                k as f32
            };
        }
        let (tiles_x, tiles_y) = (layout.cols.div_ceil(TILE), layout.rows.div_ceil(TILE));
        let kw = tiles_x * TILE;
        Renderer {
            layout,
            atlas,
            max_key: (nkeys - 1) as f32,
            canon,
            keys: vec![UNKNOWN; kw * tiles_y * TILE],
            kw,
            tiles_x,
            tiles_y,
            states: Vec::new(),
            damage: vec![NO_BOX; (kw / DBLK) * tiles_y * (TILE / DBLK)],
            dbw: kw / DBLK,
            damaged: Vec::new(),
            merge_slot: Vec::new(),
            any_damage: false,
        }
    }

    /// The whole atlas as an image: one row per glyph, LEVELS columns.
    pub fn atlas_image(&self) -> (usize, usize, Vec<u32>) {
        let cs = self.layout.cell;
        let rows = self.atlas.len() / (cs * cs * LEVELS);
        let (w, h) = (LEVELS * cs, rows * cs);
        let mut img = vec![0u32; w * h];
        for key in 0..rows * LEVELS {
            let (gx, gy) = ((key % LEVELS) * cs, (key / LEVELS) * cs);
            for y in 0..cs {
                let s = key * cs * cs + y * cs;
                img[(gy + y) * w + gx..(gy + y) * w + gx + cs]
                    .copy_from_slice(&self.atlas[s..s + cs]);
            }
        }
        (w, h, img)
    }

    /// Registers a new (blank) buffer; returns its state index.
    pub fn add_buffer(&mut self) -> usize {
        let n = self.keys.len();
        self.states.push(BufState {
            shown: vec![UNKNOWN; n],
            dirty: vec![true; self.tiles_x * self.tiles_y],
        });
        self.states.len() - 1
    }

    pub fn has_damage(&self) -> bool {
        self.any_damage
    }

    /// Re-quantizes tiles the sim touched; marks changes dirty + damaged.
    pub fn sync(&mut self, sim: &mut Sim) {
        let level = sim.level();
        dispatch!(level, simd => sync_impl(simd, self, sim));
    }

    /// Brings buffer `state` up to date in `canvas` (XRGB8888, `stride` bytes).
    pub fn paint(
        &mut self,
        level: fearless_simd::Level,
        state: usize,
        canvas: &mut [u8],
        stride: usize,
    ) {
        // SAFETY: shm slots are page aligned; u32 has no invalid bit patterns.
        let (pre, px, _) = unsafe { canvas.align_to_mut::<u32>() };
        assert!(pre.is_empty());
        dispatch!(level, simd => paint_impl(simd, self, state, px, stride / 4));
    }

    /// Takes pending damage as pixel rects (x, y, w, h), at most MAX_RECTS.
    pub fn take_damage(&mut self) -> Vec<(i32, i32, i32, i32)> {
        let union = |a: [u16; 4], b: [u16; 4]| {
            [
                a[0].min(b[0]),
                a[1].min(b[1]),
                a[2].max(b[2]),
                a[3].max(b[3]),
            ]
        };
        // (block x, block y, box) at the current granularity.
        let mut boxes: Vec<(usize, usize, [u16; 4])> = Vec::with_capacity(self.damaged.len());
        for i in self.damaged.drain(..) {
            let i = i as usize;
            let b = std::mem::replace(&mut self.damage[i], NO_BOX);
            boxes.push((i % self.dbw, i / self.dbw, b));
        }
        self.any_damage = false;
        // Too many rects: merge 2x2 groups of boxes until under the cap.
        let mut gw = self.dbw;
        while boxes.len() > MAX_RECTS {
            gw = gw.div_ceil(2);
            let slot = &mut self.merge_slot;
            slot.clear();
            slot.resize(
                gw * (self.damage.len() / self.dbw).div_ceil(2).max(1),
                u32::MAX,
            );
            let mut merged: Vec<(usize, usize, [u16; 4])> = Vec::with_capacity(boxes.len() / 2);
            for (x, y, b) in boxes {
                let (x, y) = (x / 2, y / 2);
                let s = &mut slot[y * gw + x];
                if *s == u32::MAX {
                    *s = merged.len() as u32;
                    merged.push((x, y, b));
                } else {
                    let m = &mut merged[*s as usize].2;
                    *m = union(*m, b);
                }
            }
            boxes = merged; // terminates: coordinates halve toward one box
        }
        let l = &self.layout;
        boxes
            .into_iter()
            .map(|(_, _, [x0, y0, x1, y1])| {
                let (px0, py0, _, _) = l.cell_rect(x0 as usize, y0 as usize);
                let (_, _, px1, py1) = l.cell_rect(x1 as usize - 1, y1 as usize - 1);
                (
                    px0 as i32,
                    py0 as i32,
                    (px1 - px0) as i32,
                    (py1 - py0) as i32,
                )
            })
            .collect()
    }
}

#[simd]
fn sync_impl<S: Simd>(simd: S, r: &mut Renderer, sim: &mut Sim) {
    const { assert!(TILE.is_multiple_of(8)) };
    let (cols, rows, kw) = (r.layout.cols, r.layout.rows, r.kw);
    let zero = f32x8::splat(simd, 0.0);
    let lv = f32x8::splat(simd, LEVELS as f32);
    let maxk = f32x8::splat(simd, r.max_key);
    for t in 0..sim.touched.len() {
        if !std::mem::take(&mut sim.touched[t]) {
            continue;
        }
        let (tx, ty) = (t % sim.tiles_x, t / sim.tiles_x);
        if tx >= r.tiles_x || ty >= r.tiles_y {
            continue; // padding-only tile
        }
        let x0 = tx * TILE;
        let mut changed = false;
        for y in ty * TILE..((ty + 1) * TILE).min(rows) {
            let src = &sim.dye_row(y)[x0..x0 + TILE];
            let dst = &mut r.keys[y * kw + x0..y * kw + x0 + TILE];
            for x in (0..TILE).step_by(8) {
                // key = floor(v * LEVELS) = glyph * LEVELS + step (tone/morph).
                let v = f32x8::from_slice(simd, &src[x..x + 8]);
                let k = (v * lv).floor().max(zero).min(maxk);
                let old = f32x8::from_slice(simd, &dst[x..x + 8]);
                if !k.simd_ne(old).any_true() {
                    continue;
                }
                // Rare path (cells changed): canonicalize changed lanes only.
                let mut kk = [0.0; 8];
                k.store_slice(&mut kk);
                let mut bits = 0u32;
                let mut m = k.simd_ne(old).to_bitmask() as u32;
                while m != 0 {
                    let i = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let c = r.canon[kk[i] as usize];
                    if c != dst[x + i] {
                        dst[x + i] = c;
                        bits |= 1 << i;
                    }
                }
                if bits == 0 {
                    continue;
                }
                // Extend the damage box of each 4-cell half with changes.
                for half in 0..2 {
                    let nib = (bits >> (half * DBLK)) & 0xF;
                    if nib == 0 {
                        continue;
                    }
                    let b0 = x0 + x + half * DBLK;
                    let lo = b0 + nib.trailing_zeros() as usize;
                    let hi = (b0 + 32 - nib.leading_zeros() as usize).min(cols);
                    if lo >= hi {
                        continue; // padding cells only
                    }
                    let bi = (y / DBLK) * r.dbw + b0 / DBLK;
                    let d = &mut r.damage[bi];
                    if d[0] == NO_BOX[0] {
                        r.damaged.push(bi as u32);
                    }
                    *d = [
                        d[0].min(lo as u16),
                        d[1].min(y as u16),
                        d[2].max(hi as u16),
                        d[3].max(y as u16 + 1),
                    ];
                    changed = true;
                }
            }
        }
        if changed {
            let ti = ty * r.tiles_x + tx;
            r.any_damage = true;
            for st in &mut r.states {
                st.dirty[ti] = true;
            }
        }
    }
}

/// Copies a short pixel row with inline 8-lane moves (last chunk overlaps,
/// no scalar tail); avoids a `memmove` call per short row.
#[inline(always)]
fn copy_row<S: Simd>(simd: S, dst: &mut [u32], src: &[u32]) {
    let n = dst.len();
    let src = &src[..n];
    if n < 8 {
        dst.copy_from_slice(src);
        return;
    }
    for (d, s) in dst
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<8>().0)
    {
        u32x8::from_slice(simd, s).store_slice(d);
    }
    u32x8::from_slice(simd, &src[n - 8..]).store_slice(&mut dst[n - 8..]);
}

#[simd]
fn paint_impl<S: Simd>(simd: S, r: &mut Renderer, state: usize, px: &mut [u32], stride: usize) {
    let st = &mut r.states[state];
    let (l, cs) = (&r.layout, r.layout.cell);
    for t in 0..st.dirty.len() {
        if !std::mem::take(&mut st.dirty[t]) {
            continue;
        }
        let (tx, ty) = (t % r.tiles_x, t / r.tiles_x);
        for j in ty * TILE..((ty + 1) * TILE).min(l.rows) {
            for i in tx * TILE..((tx + 1) * TILE).min(l.cols) {
                let c = j * r.kw + i;
                let k = r.keys[c];
                if st.shown[c] == k {
                    continue;
                }
                st.shown[c] = k;
                let (x0, y0, x1, y1) = l.cell_rect(i, j);
                // Sprite offset when the cell is clipped at the top/left edge.
                let sx = (x0 as isize - (l.ox + (i * cs) as isize)) as usize;
                let sy = (y0 as isize - (l.oy + (j * cs) as isize)) as usize;
                let sprite = &r.atlas[k as usize * cs * cs..];
                for y in y0..y1 {
                    let s = (sy + y - y0) * cs + sx;
                    copy_row(
                        simd,
                        &mut px[y * stride + x0..y * stride + x1],
                        &sprite[s..s + x1 - x0],
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Logical x of the left edge of global column `g` as `l` draws it.
    fn left_edge(l: &Layout, origin: i32, pos: i32, scale: f64, g: i32) -> f64 {
        f64::from(pos) + (l.ox + (g - origin) as isize * l.cell as isize) as f64 / scale
    }

    #[test]
    fn spanned_grid_continues_across_outputs() {
        // eDP-1 2560x1600 @2 at (0,0) and DP-9 1920x1080 @1 at (1280,0);
        // cells sized by the taller output (1080 logical px / 52 rows).
        let lcell = 1080.0 / 52.0;
        let (a, ga) = Layout::spanned(2560, 1600, 2.0, lcell, (0, 0));
        let (b, gb) = Layout::spanned(1920, 1080, 1.0, lcell, (1280, 0));
        assert_eq!(ga, (0, 0));
        assert_eq!((a.ox, a.oy), (0, 0));
        // Same logical cell size on both despite the different scales.
        assert_eq!(a.cell as f64 / 2.0, b.cell as f64);
        // The column cut by the seam starts at the same desktop x on both.
        let seam = gb.0;
        assert!(seam > 0 && b.ox < 0, "seam column straddles the edge");
        let (xa, xb) = (
            left_edge(&a, ga.0, 0, 2.0, seam),
            left_edge(&b, gb.0, 1280, 1.0, seam),
        );
        assert!((xa - xb).abs() < 0.5, "{xa} vs {xb}");
        // And a desktop point maps to the same global cell on both sides.
        let ca = a.to_cell((1279.9 - 0.0) * 2.0, 0.0).0 + ga.0 as f32;
        let cb = b.to_cell(-0.1, 0.0).0 + gb.0 as f32;
        assert_eq!(ca.floor(), cb.floor());
    }

    #[test]
    fn spanned_layout_covers_the_buffer() {
        for (pos, scale) in [((0, 0), 1.0), ((1280, 37), 1.5), ((-900, -5), 2.0)] {
            let (l, _) = Layout::spanned(1000, 700, scale, 19.3, pos);
            assert!(l.ox <= 0 && l.ox > -(l.cell as isize), "{pos:?}");
            assert!(l.oy <= 0 && l.oy > -(l.cell as isize), "{pos:?}");
            assert!(l.ox + (l.cols * l.cell) as isize >= 1000);
            assert!(l.oy + (l.rows * l.cell) as isize >= 700);
            let last_col = ((l.cols - 1) * l.cell) as isize;
            assert!(l.ox + last_col < 1000, "no empty column");
        }
    }
}
