# Third-party licenses

This package incorporates code from third-party open-source projects.
Each section reproduces the license text and lists the files that
derive from it. The `Copyright (c) ... AgiBot Inc.` headers elsewhere
in this package apply to original GenieSim code; this file documents
the attribution + license terms for the derived portions.

---

## PythonRobotics

* **Project**: PythonRobotics, Atsushi Sakai and contributors
* **Upstream**: https://github.com/AtsushiSakai/PythonRobotics
* **License**: MIT
* **Files derived from this project**:
  - `genie_sim_planning/math_utils.py` — `solve_dare`, `dlqr`
    (discrete-time algebraic Riccati equation + discrete LQR, adapted
    from `PathTracking/lqr_speed_steer_control/lqr_speed_steer_control.py`)
  - `scripts/simple_follow_trajectory.py` — `lqr_steering_control`,
    `calc_nearest_index` (same upstream source)

### MIT License

```
The MIT License (MIT)

Copyright (c) 2016 - 2024 Atsushi Sakai and other contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
