"""Assumed hand load-cell geometry, separate from copied factory AI code.

Values describe lateral interference and a dimensionless contact index, not N,
bar or calibrated pressure. Actual thresholds and PLC arbitration are unknown.
"""
import math

FRACTION = .22


def contact(p, radius, drum_x, lay_x, hand_x=0.):
    dy, dz = p['guide_height']-p['shaft'], p['guide_distance']
    distance = math.hypot(dy, dz)
    def angle(r):
        return math.atan2(dz, dy)+math.acos(min(.999, r/distance))
    initial_r = (p['core']+p['cable'])/2
    hand_z = dz+FRACTION*(initial_r*math.sin(angle(initial_r))-dz)
    fraction = (hand_z-dz)/(radius*math.sin(angle(radius))-dz)
    offset = fraction*(drum_x+lay_x)-hand_x
    gap = .16*p['cable']  # Assumed lateral clearance on either side.
    excess = max(0., abs(offset)-gap)
    signal = (1 if offset > 0 else -1) if excess > 0 else 0
    index = min(1., excess/(p['cable']*.5))
    return dict(offset=offset, gap=gap, signal=signal,
                left=index if signal < 0 else 0.,
                right=index if signal > 0 else 0.)
