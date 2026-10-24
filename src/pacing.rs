//! Per-output frame pacing, free of Wayland so it can be tested with a fake
//! clock. Ticks are capped to one per `interval` and gated on frame
//! callbacks, so hidden outputs stop ticking; once the sim settles nothing
//! is scheduled until input wakes it. In [`Rate::Display`] mode every frame
//! callback ticks right away and `interval` is the refresh period. Every
//! method returns the timer the caller should arm, if any (`Some(wait)`);
//! at most one is armed at a time.

use std::time::{Duration, Instant};

/// Retry delay after a failed present (shm allocation, attach).
pub const RETRY: Duration = Duration::from_secs(1);
/// Longest sim step after a stall (a suspended laptop must not explode).
const MAX_DT: f32 = 0.1;

/// Outcome of presenting a frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Present {
    /// A buffer was committed.
    Committed,
    /// Nothing changed on screen.
    Skipped,
    /// Every buffer is held by the compositor.
    Busy,
    /// An shm error (already logged).
    Failed,
}

/// How fast an animating output ticks.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Rate {
    /// At most this many frames per second.
    Fps(f32),
    /// Once per display refresh, paced by frame callbacks.
    Display,
}

impl Rate {
    /// Tick interval on an output refreshing at `refresh_mhz` (0 = unknown).
    pub fn interval(self, refresh_mhz: i32) -> Duration {
        let hz = match self {
            Rate::Fps(f) => f,
            Rate::Display if refresh_mhz > 0 => refresh_mhz as f32 / 1000.0,
            Rate::Display => 60.0,
        };
        Duration::from_secs_f32(1.0 / hz.max(1.0))
    }
}

#[derive(Debug)]
pub struct Pacer {
    interval: Duration,
    /// [`Rate::Display`]: tick on every frame callback.
    follow: bool,
    /// A frame callback is outstanding: the compositor will tell us when to
    /// draw next, and won't if the output is hidden.
    frame_pending: bool,
    /// A timer is armed (its token lives with the caller).
    timer: bool,
    /// Start of the last tick.
    last: Instant,
    /// The last tick left the sim settled.
    idle: bool,
}

impl Pacer {
    pub fn new(rate: Rate, refresh_mhz: i32, now: Instant) -> Self {
        Pacer {
            interval: rate.interval(refresh_mhz),
            follow: rate == Rate::Display,
            frame_pending: false,
            timer: false,
            last: now,
            idle: true,
        }
    }

    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// The output's refresh rate changed.
    pub fn set_refresh(&mut self, rate: Rate, refresh_mhz: i32) {
        self.interval = rate.interval(refresh_mhz);
        self.follow = rate == Rate::Display;
    }

    /// Input (or a setting change) wants a frame.
    pub fn wake(&mut self, now: Instant) -> Option<Duration> {
        self.arm(self.wait(now))
    }

    /// A present outside a tick (first frame of a new view) failed.
    pub fn retry(&mut self) -> Option<Duration> {
        self.arm(RETRY)
    }

    /// The caller could not arm the timer it was told to.
    pub fn disarm(&mut self) {
        self.timer = false;
    }

    /// The armed timer fired: a tick starts. Returns the sim step, seconds.
    pub fn tick(&mut self, now: Instant) -> f32 {
        self.timer = false;
        let dt = if self.idle {
            // Waking from rest: the gap since the last tick isn't sim time.
            self.interval.as_secs_f32()
        } else {
            (now - self.last).as_secs_f32().min(MAX_DT)
        };
        self.last = now;
        dt
    }

    /// A buffer was committed together with a frame callback request.
    pub fn frame_requested(&mut self) {
        self.frame_pending = true;
    }

    /// End of a tick: `present` is what drawing did, `active` whether the
    /// sim still moves.
    pub fn presented(&mut self, present: Present, active: bool, now: Instant) -> Option<Duration> {
        self.idle = !active;
        match present {
            // The frame callback schedules the next tick.
            Present::Committed => None,
            // Out of memory or similar: don't spin at the frame cap.
            Present::Failed => self.arm(RETRY),
            // The screen is stale until a buffer frees up. Wait out the cap
            // even when settled, or this would spin until the release.
            Present::Busy => self.arm(self.until_cap(now)),
            Present::Skipped if active => self.arm(self.wait(now)),
            Present::Skipped => None,
        }
    }

    /// The compositor's frame callback: a good time to draw the next frame.
    pub fn on_frame(&mut self, active: bool, now: Instant) -> Option<Duration> {
        self.frame_pending = false;
        if !active {
            return None;
        }
        // The compositor paces us: drawing now makes the next refresh.
        let wait = if self.follow {
            Duration::ZERO
        } else {
            self.wait(now)
        };
        self.arm(wait)
    }

    /// Next tick: right away when waking from rest, else at the cap.
    fn wait(&self, now: Instant) -> Duration {
        if self.idle {
            Duration::ZERO
        } else {
            self.until_cap(now)
        }
    }

    fn until_cap(&self, now: Instant) -> Duration {
        self.interval.saturating_sub(now - self.last)
    }

    fn arm(&mut self, wait: Duration) -> Option<Duration> {
        if self.frame_pending || self.timer {
            return None;
        }
        self.timer = true;
        Some(wait)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: Duration = Duration::from_millis(1);
    const FRAME: Duration = Duration::from_millis(20);

    fn pacer() -> (Pacer, Instant) {
        let t0 = Instant::now();
        (Pacer::new(Rate::Fps(50.0), 0, t0), t0)
    }

    #[test]
    fn rate_intervals() {
        let hz = |d: Duration| (1.0 / d.as_secs_f64()).round();
        assert_eq!(hz(Rate::Fps(60.0).interval(165_001)), 60.0);
        assert_eq!(hz(Rate::Display.interval(165_001)), 165.0);
        assert_eq!(hz(Rate::Display.interval(0)), 60.0, "unknown refresh");
    }

    #[test]
    fn display_rate_ticks_on_every_frame_callback() {
        let t0 = Instant::now();
        let mut p = Pacer::new(Rate::Display, 165_000, t0);
        p.wake(t0);
        p.tick(t0);
        p.frame_requested();
        p.presented(Present::Committed, true, t0);
        assert_eq!(p.on_frame(true, t0 + MS), Some(Duration::ZERO));
        // Without commits (invisible motion), tick at the refresh period.
        p.tick(t0 + MS);
        let wait = p.presented(Present::Skipped, true, t0 + MS).unwrap();
        assert!((6000..6100).contains(&wait.as_micros()), "{wait:?}");
    }

    #[test]
    fn refresh_change_updates_the_cap() {
        let t0 = Instant::now();
        let mut p = Pacer::new(Rate::Display, 165_000, t0);
        p.set_refresh(Rate::Display, 60_000);
        assert_eq!((1.0 / p.interval().as_secs_f64()).round(), 60.0);
        p.set_refresh(Rate::Fps(30.0), 165_000);
        assert_eq!((1.0 / p.interval().as_secs_f64()).round(), 30.0);
    }

    #[test]
    fn wakes_from_rest_immediately_with_a_nominal_step() {
        let (mut p, t0) = pacer();
        let t = t0 + Duration::from_secs(30);
        assert_eq!(p.wake(t), Some(Duration::ZERO));
        assert_eq!(p.tick(t), FRAME.as_secs_f32());
    }

    #[test]
    fn one_timer_at_a_time() {
        let (mut p, t0) = pacer();
        assert!(p.wake(t0).is_some());
        assert_eq!(p.wake(t0), None, "a second wake must not arm another timer");
        p.tick(t0);
        assert!(p.wake(t0).is_some());
    }

    #[test]
    fn disarm_allows_rearming() {
        let (mut p, t0) = pacer();
        assert!(p.wake(t0).is_some());
        p.disarm();
        assert!(p.wake(t0).is_some());
    }

    #[test]
    fn commit_waits_for_the_frame_callback_then_the_cap() {
        let (mut p, t0) = pacer();
        p.wake(t0);
        p.tick(t0);
        p.frame_requested();
        assert_eq!(p.presented(Present::Committed, true, t0 + MS), None);
        // Input while the frame is pending is picked up by the callback.
        assert_eq!(p.wake(t0 + 2 * MS), None);
        assert_eq!(p.on_frame(true, t0 + 5 * MS), Some(15 * MS));
    }

    #[test]
    fn hidden_output_never_ticks() {
        let (mut p, t0) = pacer();
        p.wake(t0);
        p.tick(t0);
        p.frame_requested();
        p.presented(Present::Committed, true, t0);
        // No frame callback ever arrives; input must not start timers.
        for k in 1..100 {
            assert_eq!(p.wake(t0 + k * FRAME), None);
        }
    }

    #[test]
    fn settles_to_no_timers() {
        let (mut p, t0) = pacer();
        p.wake(t0);
        p.tick(t0);
        p.frame_requested();
        p.presented(Present::Committed, false, t0);
        assert_eq!(p.on_frame(false, t0 + FRAME), None);
        assert_eq!(p.presented(Present::Skipped, false, t0), None);
    }

    #[test]
    fn invisible_motion_still_ticks_at_the_cap() {
        let (mut p, t0) = pacer();
        p.wake(t0);
        p.tick(t0);
        assert_eq!(
            p.presented(Present::Skipped, true, t0 + 4 * MS),
            Some(16 * MS)
        );
    }

    #[test]
    fn busy_retries_at_the_cap_and_failure_backs_off() {
        let (mut p, t0) = pacer();
        p.wake(t0);
        p.tick(t0);
        // Settled (`active` false) but stale: must not retry with zero wait.
        assert_eq!(p.presented(Present::Busy, false, t0), Some(FRAME));
        p.tick(t0 + FRAME);
        assert_eq!(p.presented(Present::Failed, true, t0 + FRAME), Some(RETRY));
        assert_eq!(p.retry(), None, "already armed");
    }

    #[test]
    fn step_is_capped_after_a_stall() {
        let (mut p, t0) = pacer();
        p.wake(t0);
        p.tick(t0);
        p.presented(Present::Skipped, true, t0);
        assert_eq!(p.tick(t0 + Duration::from_secs(5)), MAX_DT);
    }
}
