//! Palette from one seed color: a three-stop OKLCH ramp (dim -> bright) at
//! the seed's hue, so accents stay monochromatic. Near-gray seeds produce a
//! neutral ramp. Interpolation happens in OKLCH; out-of-gamut colors are
//! brought in by reducing chroma, never by shifting hue.

#![allow(clippy::excessive_precision)]

/// Default OKLab lightness of the dimmest glyph tone.
pub const DIM: f32 = 0.40;
/// Lightness of the middle stop; `dim` stays below it.
const MID: f32 = 0.68;

pub struct Palette {
    pub bg: [u8; 3],
    /// (OKLab lightness, OKLCH chroma, hue°), from dim to bright.
    stops: [[f32; 3]; 3],
}

// sRGB -> linear RGB.
pub(crate) fn to_lin(c: u8) -> f32 {
    let c = f32::from(c) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

// Linear sRGB -> sRGB.
pub(crate) fn to_srgb(c: f32) -> u8 {
    let c = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.max(0.0).powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0).round().clamp(0.0, 255.0) as u8
}

fn oklch_of(rgb: [u8; 3]) -> [f32; 3] {
    let [r, g, b] = rgb.map(to_lin);

    let l = (0.41222147 * r + 0.53633254 * g + 0.05144599 * b).cbrt();
    let m = (0.21190350 * r + 0.68069955 * g + 0.10739696 * b).cbrt();
    let s = (0.08830246 * r + 0.28171884 * g + 0.62997870 * b).cbrt();

    let a = 1.9779985 * l - 2.4285922 * m + 0.4505937 * s;
    let bb = 0.0259040 * l + 0.7827718 * m - 0.8086758 * s;

    let lightness = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;

    [
        lightness,
        a.hypot(bb),
        bb.atan2(a).to_degrees().rem_euclid(360.0),
    ]
}

/// OKLCH -> sRGB, reducing chroma until the color fits the gamut.
fn srgb_of([lightness, mut c, h]: [f32; 3]) -> [u8; 3] {
    let (sin, cos) = h.to_radians().sin_cos();

    loop {
        let (a, b) = (c * cos, c * sin);

        let l = (lightness + 0.3963377774 * a + 0.2158037573 * b).powi(3);
        let m = (lightness - 0.1055613458 * a - 0.0638541728 * b).powi(3);
        let s = (lightness - 0.0894841775 * a - 1.2914855480 * b).powi(3);

        let rgb = [
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ];

        if c < 1e-3 || rgb.iter().all(|v| (-1e-4..=1.0001).contains(v)) {
            return rgb.map(to_srgb);
        }

        c *= 0.92;
    }
}

impl Palette {
    /// Generic palette from any custom RGB seed and optional background.
    pub fn new(seed: [u8; 3], bg: Option<[u8; 3]>) -> Self {
        let [_, raw_chroma, hue] = oklch_of(seed);

        // Preserve neutral colors; avoid amplifying noise in near-grays.
        let chroma = if raw_chroma < 0.025 { 0.0 } else { raw_chroma };

        let colored = chroma > 0.0;

        let background = bg.unwrap_or_else(|| {
            srgb_of([
                0.09,
                if colored {
                    (chroma * 0.10).min(0.012)
                } else {
                    0.0
                },
                hue,
            ])
        });

        let shift = if colored { 3.0 } else { 0.0 };

        Self {
            bg: background,
            stops: [
                [DIM, chroma * 0.42, hue - shift],
                [MID, chroma * 0.90, hue],
                [0.92, chroma * 0.52, hue + shift],
            ],
        }
    }

    /// Sets the dimmest tone's OKLab lightness (default [`DIM`]). Lower
    /// values let clusters fade deeper into the background. Clamped to stay
    /// above the background and below the middle stop.
    pub fn with_dim(mut self, dim: f32) -> Self {
        let floor = oklch_of(self.bg)[0] + 0.02;
        self.stops[0][0] = dim.clamp(floor, MID - 0.05);
        self
    }

    /// Color at position p in 0..1 along the ramp.
    pub fn at(&self, p: f32) -> [u8; 3] {
        let p = p.clamp(0.0, 1.0) * 2.0;
        let i = (p as usize).min(1);
        let (a, b, t) = (self.stops[i], self.stops[i + 1], p - i as f32);

        let dh = (b[2] - a[2] + 540.0).rem_euclid(360.0) - 180.0;

        srgb_of([
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + dh * t,
        ])
    }
}

pub fn parse_hex(s: &str) -> Option<[u8; 3]> {
    let s = s.strip_prefix('#').unwrap_or(s);
    if s.len() != 6 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let v = u32::from_str_radix(s, 16).ok()?;
    Some([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex() {
        assert_eq!(parse_hex("#e8be7c"), Some([0xe8, 0xbe, 0x7c]));
        assert_eq!(parse_hex("2897A1"), Some([0x28, 0x97, 0xa1]));
        for bad in ["", "#fff", "+abcde", "##abcdef", "abcdefg", "zzzzzz"] {
            assert_eq!(parse_hex(bad), None, "{bad}");
        }
    }

    #[test]
    fn oklch_roundtrip() {
        for c in [
            [232, 190, 124],
            [40, 151, 161],
            [144, 144, 146],
            [255, 0, 0],
            [0, 0, 0],
        ] {
            let back = srgb_of(oklch_of(c));
            assert!(
                c.iter().zip(back).all(|(a, b)| a.abs_diff(b) <= 1),
                "{c:?} -> {back:?}"
            );
        }
    }

    fn lightness(c: [u8; 3]) -> f32 {
        oklch_of(c)[0]
    }

    #[test]
    fn ramp_brightens_and_bg_stays_darker() {
        for seed in [
            [232, 190, 124],
            [40, 151, 161],
            [216, 120, 172],
            [128, 128, 128],
        ] {
            let p = Palette::new(seed, None);
            let ls: Vec<f32> = (0..=16).map(|i| lightness(p.at(i as f32 / 16.0))).collect();
            assert!(
                ls.windows(2).all(|w| w[1] >= w[0] - 1e-3),
                "{seed:?}: {ls:?}"
            );
            assert!(
                lightness(p.bg) < ls[0],
                "{seed:?}: background brighter than ramp"
            );
        }
    }

    #[test]
    fn dim_lowers_the_dim_end_only_and_stays_above_bg() {
        let p = Palette::new([232, 190, 124], None);
        let d = Palette::new([232, 190, 124], None).with_dim(0.28);
        assert!(lightness(d.at(0.0)) < lightness(p.at(0.0)) - 0.08);
        assert_eq!(d.at(1.0), p.at(1.0));
        let floor = Palette::new([232, 190, 124], None).with_dim(0.0);
        assert!(lightness(floor.at(0.0)) > lightness(floor.bg));
    }

    #[test]
    fn gray_seed_gives_neutral_ramp() {
        let p = Palette::new([140, 140, 140], None);
        for i in 0..=8 {
            let [r, g, b] = p.at(i as f32 / 8.0);
            assert!(r.abs_diff(g) <= 1 && g.abs_diff(b) <= 1, "{:?}", [r, g, b]);
        }
    }
}
