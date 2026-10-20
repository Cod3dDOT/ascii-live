//! Command-line parsing (no dependencies; testable without a compositor).

use crate::Config;
use crate::color::{Palette, parse_hex};
use crate::field::Style;
use crate::sim::Params;

pub const HELP: &str = "\
situ: live ASCII fluid wallpaper for Wayland (wlr-layer-shell)

USAGE: situ [OPTIONS]

Starting a new instance replaces the one already running on this display.

  --static          draw once and take no input: no animation, no wakeups,
                    minimal memory (motion options are ignored)

Color:
  --color HEX       seed color (default #5ca9b7)
  --bg HEX          exact background (default: near-black tinted by the seed)
  --gamma F         color curve along the ramp (default 1.2)

Look:
  --ramp STR        glyph ramp, darkest first, ' ' = blank (default ' .-+ox')
  --rows N          cell rows per screen height (default 54)
  --scale F         noise feature size in cells (default 14)
  --octaves N       noise octaves (default 4)
  --warp F          domain warp, 0 blobs .. 2+ marbled (default 0.8)
  --blank F         fraction of blank cells, 0..1 (default 0.45)
  --contrast F      field contrast (default 1.3)
  --seed N          noise seed (default 7)

Motion:
  --fps F           frame cap while animating (default 30)
  --damping F       velocity decay /s (default 1.4)
  --relax S         seconds for glyphs to return home (default 1.4)
  --vorticity F     swirl strength (default 3)
  --force F         mouse force gain (default 1)
  --radius CELLS    mouse splat radius (default 3.5)
  --ambient [F]     continuous ambient flow (costs battery; strength 0.6)

Mouse: move = push fluid, left/right click = burst/implode,
       scroll = swirl, middle click = toggle ambient flow.
";

pub enum Cmd {
    Help,
    Run(Config),
}

impl Default for Config {
    fn default() -> Self {
        let seed = [0x5c, 0xa9, 0xb7];
        Config {
            style: Style {
                ramp: " .-+ox".chars().collect(),
                palette: Palette::new(seed, None),
                gamma: 1.2,
                rows: 54.0,
                scale: 14.0,
                octaves: 4,
                warp: 0.8,
                blank: 0.45,
                contrast: 1.3,
                seed: 7,
            },
            params: Params::default(),
            fps: 30.0,
            interactive: true,
        }
    }
}

pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Cmd, String> {
    let mut cfg = Config::default();
    let (style, params) = (&mut cfg.style, &mut cfg.params);
    let (mut seed_color, mut bg) = ([0x5c, 0xa9, 0xb7], None);

    let mut it = args.into_iter().peekable();
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{a} needs a value"));
        let num = |s: String| {
            s.parse::<f32>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| format!("bad number for {a}: {s}"))
        };
        let color = |s: String| parse_hex(&s).ok_or_else(|| format!("bad color for {a}: {s}"));
        match a.as_str() {
            "-h" | "--help" => return Ok(Cmd::Help),
            "--static" => cfg.interactive = false,
            "--color" | "--fg" => seed_color = color(val()?)?,
            "--bg" => bg = Some(color(val()?)?),
            "--ramp" => {
                style.ramp = val()?.chars().collect();
                if !(2..=16).contains(&style.ramp.len()) {
                    return Err("--ramp needs 2..=16 glyphs".into());
                }
            }
            "--gamma" => style.gamma = num(val()?)?.max(0.05),
            "--rows" => style.rows = num(val()?)?.max(1.0),
            "--scale" => style.scale = num(val()?)?,
            "--octaves" => style.octaves = num(val()?)?.clamp(1.0, 8.0) as u32,
            "--warp" => style.warp = num(val()?)?,
            "--blank" => style.blank = num(val()?)?.clamp(0.0, 1.0),
            "--contrast" => style.contrast = num(val()?)?,
            "--seed" => style.seed = num(val()?)? as u32,
            "--fps" => cfg.fps = num(val()?)?.clamp(1.0, 240.0),
            "--damping" => params.damping = num(val()?)?,
            "--relax" => params.relax = num(val()?)?.max(0.05),
            "--vorticity" => params.vorticity = num(val()?)?,
            "--force" => params.force = num(val()?)?,
            "--radius" => params.radius = num(val()?)?.max(0.5),
            "--ambient" => {
                params.ambient_on = true;
                if let Some(v) = it.next_if(|v| v.parse::<f32>().is_ok()) {
                    params.ambient = v.parse().unwrap();
                }
            }
            _ => return Err(format!("unknown option {a}; see --help")),
        }
    }
    style.palette = Palette::new(seed_color, bg);
    if !cfg.interactive {
        cfg.params.ambient_on = false;
    }
    Ok(Cmd::Run(cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Result<Config, String> {
        match parse(args.iter().map(ToString::to_string))? {
            Cmd::Run(c) => Ok(c),
            Cmd::Help => Err("help".into()),
        }
    }

    #[test]
    fn defaults() {
        let c = run(&[]).unwrap();
        assert_eq!(c.style.ramp, vec![' ', '.', '-', '+', 'o', 'x']);
        assert!(!c.params.ambient_on);
    }

    #[test]
    fn colors_and_values() {
        let c = run(&["--color", "#e8be7c", "--bg", "040302", "--rows", "40"]).unwrap();
        assert_eq!(c.style.palette.bg, [4, 3, 2]);
        assert_eq!(c.style.rows, 40.0);
    }

    #[test]
    fn ambient_optional_value() {
        let c = run(&["--ambient", "--seed", "3"]).unwrap();
        assert!(c.params.ambient_on);
        assert_eq!(c.params.ambient, Params::default().ambient);
        assert_eq!(c.style.seed, 3);
        let c = run(&["--ambient", "0.9"]).unwrap();
        assert_eq!(c.params.ambient, 0.9);
        let c = run(&["--ambient", "--static"]).unwrap();
        assert!(!c.interactive && !c.params.ambient_on);
    }

    #[test]
    fn errors() {
        assert!(run(&["--color", "red"]).is_err());
        assert!(run(&["--rows"]).is_err());
        assert!(run(&["--rows", "nan"]).is_err());
        assert!(run(&["--ramp", "x"]).is_err());
        assert!(run(&["--bogus"]).is_err());
        assert!(matches!(parse(["-h".to_string()]), Ok(Cmd::Help)));
    }
}
