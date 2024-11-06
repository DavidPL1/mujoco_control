# mujoco_ik_control

Install with
```bash
pip install .
```

then import in python with

```python
from mujoco_ik_control import *
```

## Speed test with Opspace Controller

run on `11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz`

```
py controller:
        mean: 14.97s +- 0.07s
        min: 14.83s
        max: 15.04s
c controller 1 step:
        mean: 6.20s +- 0.25s (2.42%)
        min: 5.75s (2.58%)
        max: 6.54s (2.30%)
c controller 10 step:
        mean: 5.62s +- 0.23s (2.66%)
        min: 5.34s (2.78%)
        max: 5.95s (2.53%)
c controller 100 step:
        mean: 5.82s +- 0.02s (2.57%)
        min: 5.78s (2.57%)
        max: 5.84s (2.58%)
```

for details see [test.py](test.py#L298-L319)
