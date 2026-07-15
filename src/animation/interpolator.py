from __future__ import annotations
import numpy as np
from typing import List
from src.animation.viseme_table import VisemeTable
from src.alignment.viseme_mapper import VisemeEvent

class VisemeInterpolator:
    def __init__(self, viseme_table: VisemeTable, ramp_duration: float = 0.04):
        self.viseme_table = viseme_table
        self.ramp_duration = ramp_duration  # transition duration in seconds (default 40ms)

    def get_coefficients(self, time_s: float, timeline: List[VisemeEvent]) -> np.ndarray:
        """Interpolate viseme coefficients at a given timestamp."""
        if not timeline:
            return self.viseme_table.get_coefficients("IDLE")
        
        # 1. Edge case: Before first event
        if time_s < timeline[0].start_time:
            return self.viseme_table.get_coefficients("IDLE")
            
        # 2. Edge case: After last event
        if time_s > timeline[-1].end_time:
            return self.viseme_table.get_coefficients("IDLE")

        # 3. Find active event index
        active_idx = -1
        for i, event in enumerate(timeline):
            if event.start_time <= time_s <= event.end_time:
                active_idx = i
                break
                
        # 4. Handle gap case
        if active_idx == -1:
            for i in range(len(timeline) - 1):
                if timeline[i].end_time < time_s < timeline[i+1].start_time:
                    t_start = timeline[i].end_time
                    t_end = timeline[i+1].start_time
                    factor = (time_s - t_start) / max(t_end - t_start, 1e-5)
                    factor = max(0.0, min(1.0, factor))
                    c_prev = self.viseme_table.get_coefficients(timeline[i].name)
                    c_next = self.viseme_table.get_coefficients(timeline[i+1].name)
                    return c_prev + factor * (c_next - c_prev)
            return self.viseme_table.get_coefficients("IDLE")

        # 5. Normal case: We are inside an event
        current_event = timeline[active_idx]
        current_coeffs = self.viseme_table.get_coefficients(current_event.name)
        
        # Check if we are close to the end of the current event to start ramping
        if active_idx < len(timeline) - 1:
            next_event = timeline[active_idx + 1]
            time_to_end = current_event.end_time - time_s
            if time_to_end < self.ramp_duration:
                # Interpolate towards next event
                factor = (self.ramp_duration - time_to_end) / self.ramp_duration
                factor = max(0.0, min(1.0, factor))
                next_coeffs = self.viseme_table.get_coefficients(next_event.name)
                return current_coeffs + factor * (next_coeffs - current_coeffs)
                
        return current_coeffs
