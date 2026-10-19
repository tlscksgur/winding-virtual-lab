"""Geometric observations of the assumed plant, never factory defect grading."""
from collections import deque


class Diagnostics:
    def __init__(self, cable):
        self.cable = cable
        self.turn = deque()
        self.gap = 0.
        self.coast = 0.
        self.coast_max = 0.
        self.markers = {}

    def mark(self, key, t, value):
        self.markers[key] = dict(kind=key, t=round(t, 3), value=round(value, 3))

    def step(self, plant, before_x, dt):
        coast = abs(plant.x-before_x) if not plant.effective_relay else 0.
        self.coast = self.coast+coast if not plant.effective_relay else 0.
        if self.coast > self.coast_max+.001:
            self.coast_max = self.coast
            self.mark('coast', plant.t, self.coast_max)

    def deposition(self, plant):
        q, c, r = plant.q, plant.c, plant.radius
        # Interpolate one revolution earlier at the same drum phase. Comparing
        # adjacent samples would measure a tiny path step instead of turn pitch.
        self.turn.append((q, c, r, plant.layer, plant.climbing))
        target = q-1
        while len(self.turn) > 2 and self.turn[1][0] <= target:
            self.turn.popleft()
        a = self.turn[0]
        b = self.turn[1] if len(self.turn) > 1 else a
        if (target < a[0] or plant.climbing or a[4] or b[4]
                or a[3] != plant.layer or b[3] != plant.layer):
            return
        u = (target-a[0])/(b[0]-a[0]) if b[0] > a[0] else 0
        previous_c = a[1]+(b[1]-a[1])*u
        gap = max(0., abs(c-previous_c)-self.cable)
        if gap > self.gap+.001:
            self.gap = gap
            self.mark('gap', plant.t, gap)

    def finish(self, plant, trace):
        for key, predicate in (
            ('contact', lambda s: s['lc_signal'] != 0 and plant.p['loadcell'] != 'off'),
            ('conflict', lambda s: plant.p['loadcell'] == 'underroller' and
             s['lc_drive'] and s['relay'] and -s['lc_drive'] != s['relay']),
            ('edge', lambda s: s['edge'] > 1.1),
        ):
            found = next((s for s in trace if predicate(s)), None)
            if found:
                self.mark(key, found['t'], found['edge'] if key == 'edge' else 1)
        if plant.missed_pulses:
            self.mark('missed', plant.first_missed, plant.missed_pulses)
        return dict(max_turn_gap_mm=round(self.gap, 3),
                    max_coast_mm=round(self.coast_max, 3),
                    missed_pulses=plant.missed_pulses,
                    delivered_pulses=plant.delivered_pulses), list(self.markers.values())
