# mujoco_ik_control

## Install (Linux w conda)

setup conda environment

1. Set Mujoco version (tested with MuJoCo 3.2.3)
```bash
export MUJOCO_VERSION=3.2.3
```

2. Install mujoco
```bash
mkdir ${HOME}/.mujoco
cd ${HOME}/.mujoco
wget https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz
tar -xzf mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz && rm mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz
pip install mujoco==${MUJOCO_VERSION}
```

3. Create utility script for setting up paths
```bash
echo "
# For cmake to find mujoco
export MUJOCO_DIR=\${HOME}/volmounts/mujoco/mujoco-\${MUJOCO_VERSION}
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\${MUJOCO_DIR}/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia
# for offscreen rendering
export MUJOCO_GL=egl

export CUDA_HOME=\$CONDA_PREFIX
" > ${HOME}/sourcescript.sh
```

4. Clone this repository
```bash
git clone https://github.com/DavidPL1/mujoco_ik_control.git ${HOME}/mujoco_ik_control
```

5. Install library with bindings in conda
```bash
source ${HOME}/sourcescript.sh
pip install ${HOME}/mujoco_ik_control
```

## Usage

### Opspace Controller
import in python with

```python
from mujoco_ik_control import opspace_ctrl

# initialize model and data, and compute necessary data

...

# instanciate controller
ctrlr = opspace_ctrl.OpspaceController(
        model,
        site_id,
        dof_ids,
        actuator_ids,
        True,  # gravcomp
        q0,
        integration_dt,
        Kpos,
        Kori,
        Kp,
        Kd,
        Kp_null,
        Kd_null,
    )

...

# run controller steps

ctrlr.run_steps(data, step_size, target_pos, target_quat)

```


### Speed test

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
