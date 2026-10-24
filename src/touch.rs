//! Touch gestures, free of Wayland for testing: each touch point drags the
//! fluid like the pointer, and a short touch that barely moves is a tap
//! (a burst, like a left click). Points are keyed by `K` (the surface) so
//! a point keeps feeding the output it went down on.

/// Max travel (cells) and duration (ms) for a touch to count as a tap.
const TAP_TRAVEL: f32 = 1.0;
const TAP_MS: u32 = 300;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Gesture {
    /// Dragged from `(x0, y0)` to `(x1, y1)`, in cells.
    Push(f32, f32, f32, f32),
    Tap(f32, f32),
}

struct Point<K> {
    id: i32,
    key: K,
    last: (f32, f32),
    travel: f32,
    down_ms: u32,
}

pub struct Touches<K> {
    points: Vec<Point<K>>,
}

impl<K> Default for Touches<K> {
    fn default() -> Self {
        Touches { points: Vec::new() }
    }
}

impl<K: Clone> Touches<K> {
    /// The surface touch point `id` went down on.
    pub fn key(&self, id: i32) -> Option<&K> {
        self.find(id).map(|p| &p.key)
    }

    pub fn down(&mut self, id: i32, key: K, pos: (f32, f32), time_ms: u32) {
        self.points.retain(|p| p.id != id);
        self.points.push(Point {
            id,
            key,
            last: pos,
            travel: 0.0,
            down_ms: time_ms,
        });
    }

    pub fn motion(&mut self, id: i32, pos: (f32, f32)) -> Option<(K, Gesture)> {
        let p = self.points.iter_mut().find(|p| p.id == id)?;
        let (x0, y0) = std::mem::replace(&mut p.last, pos);
        p.travel += (pos.0 - x0).hypot(pos.1 - y0);
        Some((p.key.clone(), Gesture::Push(x0, y0, pos.0, pos.1)))
    }

    pub fn up(&mut self, id: i32, time_ms: u32) -> Option<(K, Gesture)> {
        let i = self.points.iter().position(|p| p.id == id)?;
        let p = self.points.swap_remove(i);
        let quick = time_ms.wrapping_sub(p.down_ms) <= TAP_MS;
        (quick && p.travel <= TAP_TRAVEL).then_some((p.key, Gesture::Tap(p.last.0, p.last.1)))
    }

    /// The compositor took the sequence (e.g. for a gesture): forget it.
    pub fn cancel(&mut self) {
        self.points.clear();
    }

    fn find(&self, id: i32) -> Option<&Point<K>> {
        self.points.iter().find(|p| p.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drag_pushes_from_the_last_position() {
        let mut t = Touches::default();
        t.down(0, 'a', (1.0, 1.0), 0);
        assert_eq!(
            t.motion(0, (3.0, 1.0)),
            Some(('a', Gesture::Push(1.0, 1.0, 3.0, 1.0)))
        );
        assert_eq!(
            t.motion(0, (3.0, 4.0)),
            Some(('a', Gesture::Push(3.0, 1.0, 3.0, 4.0)))
        );
        assert_eq!(t.up(0, 100), None, "a drag is not a tap");
        assert_eq!(t.motion(0, (5.0, 5.0)), None, "point is gone");
    }

    #[test]
    fn taps_need_to_be_short_and_still() {
        let mut t = Touches::default();
        t.down(1, 'a', (2.0, 2.0), 1000);
        t.motion(1, (2.4, 2.3));
        assert_eq!(t.up(1, 1200), Some(('a', Gesture::Tap(2.4, 2.3))));
        t.down(1, 'a', (2.0, 2.0), 1000);
        assert_eq!(t.up(1, 1000 + TAP_MS + 1), None, "long press");
        // The millisecond clock may wrap.
        t.down(1, 'a', (2.0, 2.0), u32::MAX - 10);
        assert!(t.up(1, 50).is_some());
    }

    #[test]
    fn multitouch_points_are_independent() {
        let mut t = Touches::default();
        t.down(0, 'a', (0.0, 0.0), 0);
        t.down(1, 'b', (9.0, 9.0), 0);
        assert_eq!(t.key(1), Some(&'b'));
        assert_eq!(
            t.motion(1, (9.0, 8.0)),
            Some(('b', Gesture::Push(9.0, 9.0, 9.0, 8.0)))
        );
        assert!(t.up(0, 10).is_some());
        assert_eq!(t.key(0), None);
        assert_eq!(t.key(1), Some(&'b'));
        t.cancel();
        assert_eq!(t.key(1), None);
        assert_eq!(t.up(1, 10), None, "cancelled points don't tap");
    }

    #[test]
    fn reused_id_replaces_the_stale_point() {
        let mut t = Touches::default();
        t.down(0, 'a', (0.0, 0.0), 0);
        t.down(0, 'b', (5.0, 5.0), 0);
        assert_eq!(t.key(0), Some(&'b'));
        assert_eq!(
            t.motion(0, (6.0, 5.0)),
            Some(('b', Gesture::Push(5.0, 5.0, 6.0, 5.0)))
        );
    }
}
