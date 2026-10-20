//! Process lifecycle: one wallpaper per Wayland display, and never linger
//! in a stopped state while owning layer surfaces.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Holds the per-display lock for the life of the process.
pub struct Instance(#[expect(dead_code, reason = "held for its lock")] File);

fn lock_path() -> Result<PathBuf, String> {
    let dir = std::env::var_os("XDG_RUNTIME_DIR").ok_or("XDG_RUNTIME_DIR is not set")?;
    let display = std::env::var("WAYLAND_DISPLAY").unwrap_or_else(|_| "wayland-0".into());
    Ok(PathBuf::from(dir).join(format!("situ-{}.lock", display.replace('/', "_"))))
}

fn signal_pid(pid: i32, sig: i32) {
    // SAFETY: plain syscall; pid > 0 targets a single process.
    unsafe { libc::kill(pid, sig) };
}

/// Becomes the wallpaper for this display, replacing a running instance
/// (SIGTERM, then SIGKILL after a grace period).
pub fn take_over() -> Result<Instance, String> {
    let path = lock_path()?;
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(err)?;

    let start = Instant::now();
    let mut pid = None;
    loop {
        match f.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::Error(e)) => return Err(err(e)),
            Err(std::fs::TryLockError::WouldBlock) => {}
        }
        if pid.is_none() {
            let mut s = String::new();
            f.rewind()
                .and_then(|_| f.read_to_string(&mut s))
                .map_err(err)?;
            pid = s.trim().parse::<i32>().ok().filter(|&p| p > 0);
            if let Some(p) = pid {
                // SIGCONT so a stopped holder can act on SIGTERM.
                signal_pid(p, libc::SIGTERM);
                signal_pid(p, libc::SIGCONT);
            }
        }
        if start.elapsed() > Duration::from_secs(2) {
            if let Some(p) = pid {
                signal_pid(p, libc::SIGKILL);
            }
            f.lock().map_err(err)?;
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    f.set_len(0).map_err(err)?;
    f.rewind().map_err(err)?;
    write!(f, "{}", std::process::id()).map_err(err)?;
    Ok(Instance(f))
}

/// Ctrl+Z would freeze the process while it still owns its layer surfaces,
/// leaving a dead wallpaper on screen. Exit instead so the compositor drops
/// them.
pub fn exit_on_suspend() {
    extern "C" fn quit(_: libc::c_int) {
        // SAFETY: `_exit` is async-signal-safe.
        unsafe { libc::_exit(0) }
    }
    // SAFETY: installs an async-signal-safe handler.
    unsafe {
        libc::signal(
            libc::SIGTSTP,
            quit as extern "C" fn(libc::c_int) as libc::sighandler_t,
        )
    };
}
