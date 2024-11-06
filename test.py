from mujoco_ik_control import opspace_ctrl

import mujoco
import mujoco.viewer
import numpy as np

import pdb

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.4f}".format})

import time

impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95

# Integration timestep in seconds.
integration_dt: float = 0.2

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002


# Opspace controller taken from https://github.com/kevinzakka/mjctrl/blob/main/opspace.py
# and adapted to Panda
def py_control(total_steps=None) -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("../assets/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)

    model.opt.timestep = dt

    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    # End-effector site we wish to control.
    site_name = "S_eef"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    act_names = [
        "actuator1",
        "actuator2",
        "actuator3",
        "actuator4",
        "actuator5",
        "actuator6",
        "actuator7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in act_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos[:-2]

    # Mocap body we will control with our mouse.
    mocap_name = "target_mocap"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Pre-allocate numpy arrays.
    fjac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    Mx = np.zeros((6, 6))

    pos = np.array([0.5, 0.0, 0.4])
    quat = np.array([0.0, 1.0, 0.0, 0.0])

    # with mujoco.viewer.launch_passive(
    #     model=model,
    #     data=data,
    #     show_left_ui=False,
    #     show_right_ui=False,
    # ) as viewer:
    # Reset the simulation.
    # mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Reset the free camera.
    # mujoco.mjv_defaultFreeCamera(model, viewer.cam)

    # Enable site frame visualization.
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    remaining_steps = total_steps
    # while viewer.is_running() and (total_steps is None or remaining_steps > 0):
    # step_start = time.time()
    while total_steps is None or remaining_steps > 0:

        pos = data.mocap_pos[mocap_id]
        quat = data.mocap_quat[mocap_id]

        # Spatial velocity (aka twist).
        # dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
        dx = pos - data.site(site_id).xpos
        twist[:3] = Kpos * dx / integration_dt
        mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, quat, site_quat_conj)
        mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
        twist[3:] *= Kori / integration_dt

        # Jacobian.
        mujoco.mj_jacSite(model, data, fjac[:3], fjac[3:], site_id)
        jac = fjac[:, :7]

        # Compute the task-space inertia matrix.
        mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))

        Mx_inv = jac @ M_inv[:-2, :-2] @ jac.T
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        # Compute generalized forces.
        tau = jac.T @ Mx @ (Kp * twist - Kd * (jac @ data.qvel[dof_ids]))

        # Add joint task in nullspace.
        Jbar = M_inv[:-2, :-2] @ jac.T @ Mx
        ddq = Kp_null * (q0 - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids]
        tau += (np.eye(model.nv - 2) - jac.T @ Jbar.T) @ ddq

        # Add gravity compensation.
        if gravity_compensation:
            tau += data.qfrc_bias[dof_ids]

        # Set the control signal and step the simulation.
        np.clip(tau, *model.actuator_ctrlrange[actuator_ids].T, out=tau)

        data.ctrl[actuator_ids] = tau[actuator_ids]
        mujoco.mj_step(model, data)

        if total_steps is not None:
            remaining_steps -= 1

            # viewer.sync()
            # time_until_next_step = dt - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)


def c_ext(total_steps=None, step_size=1) -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("../assets/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)

    # print(f"nv: {model.nv}")

    model.opt.timestep = dt
    # dt = 0.002

    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    # End-effector site we wish to control.
    site_name = "S_eef"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    act_names = [
        "actuator1",
        "actuator2",
        "actuator3",
        "actuator4",
        "actuator5",
        "actuator6",
        "actuator7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in act_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos[:-2]

    # Mocap body we will control with our mouse.
    mocap_name = "target_mocap"
    mocap_id = model.body(mocap_name).mocapid[0]

    pos = np.array([0.5, 0.0, 0.4])
    quat = np.array([0.0, 1.0, 0.0, 0.0])

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

    remaining_steps = total_steps

    # with mujoco.viewer.launch_passive(
    #     model=model,
    #     data=data,
    #     show_left_ui=False,
    #     show_right_ui=False,
    # ) as viewer:
    # Reset the simulation.
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Reset the free camera.
    # mujoco.mjv_defaultFreeCamera(model, viewer.cam)

    # Enable site frame visualization.
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    while total_steps is None or remaining_steps > 0:
        pos = data.mocap_pos[mocap_id]
        quat = data.mocap_quat[mocap_id]
        ctrlr.run_steps(data, step_size, pos, quat)

        if total_steps is not None:
            remaining_steps -= step_size

        # while viewer.is_running() and (total_steps is None or remaining_steps > 0):
        #     step_start = time.time()
        #     pos = data.mocap_pos[mocap_id]
        #     quat = data.mocap_quat[mocap_id]
        #     ctrlr.run_steps(data, step_size, pos, quat)

        #     if total_steps is not None:
        #         remaining_steps -= step_size

        #     viewer.sync()
        #     time_until_next_step = dt - (time.time() - step_start)
        #     if time_until_next_step > 0:
        #         time.sleep(time_until_next_step)


import timeit

if __name__ == "__main__":
    breakpoint()
    t_py = timeit.Timer(lambda: py_control(1000))
    t_c1 = timeit.Timer(lambda: c_ext(1000, 1))
    t_c10 = timeit.Timer(lambda: c_ext(1000, 10))
    t_c100 = timeit.Timer(lambda: c_ext(1000, 100))

    r1 = t_py.repeat(5, 100)
    r2 = t_c1.repeat(5, 100)
    r3 = t_c10.repeat(5, 100)
    r4 = t_c100.repeat(5, 100)

    print(
        f"py controller:\n\tmean: {np.mean(r1)}+-{np.std(r1)}\nmin: {np.min(r1)}\nmax: {np.max(r1)}"
    )
    print(
        f"c controller 1 step:\n\tmean: {np.mean(r2)}+-{np.std(r2)}\nmin: {np.min(r2)}\nmax: {np.max(r2)}"
    )
    print(
        f"c controller 10 steps:\n\tmean: {np.mean(r3)}+-{np.std(r3)}\nmin: {np.min(r3)}\nmax: {np.max(r3)}"
    )
    print(
        f"c controller 100 steps:\n\tmean: {np.mean(r4)}+-{np.std(r4)}\nmin: {np.min(r4)}\nmax: {np.max(r4)}"
    )
