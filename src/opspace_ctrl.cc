#include <mujoco/mujoco.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>
#include <pybind11/stl.h>
#include <math.h>

#include <iostream>
#include <sstream>
#include <optional>

#include <Eigen/Dense>
#include <algorithm> // for std::max and std::min

namespace mujoco::python
{

    namespace
    {

        namespace py = ::pybind11;

        mjtNum *get_array_ptr(const py::array_t<mjtNum> arr, const char *name, int nstep, int ndim)
        {
            py::buffer_info info = arr.request();

            // Check expected size
            int expected_size = nstep * ndim;
            if (info.size != expected_size)
            {
                std::ostringstream msg;
                msg << name << ".size should be " << expected_size << ", got " << info.size;
                throw py::value_error(msg.str());
            }
            return static_cast<mjtNum *>(info.ptr);
        }
    } // end namespace anonymous

    class OpspaceController
    {
    public:
        OpspaceController(
            const mjModel *model,
            int site_id, int *dof_ids, int *actuator_ids, bool gravcomp, mjtNum *q0, double integration_dt,
            double Kpos, double Kori, mjtNum *Kp, mjtNum *Kd, mjtNum *Kp_null, mjtNum *Kd_null) : model(model),
                                                                                                  site_id(site_id),
                                                                                                  dof_ids(dof_ids),
                                                                                                  actuator_ids(actuator_ids),
                                                                                                  gravcomp(gravcomp),
                                                                                                  target_pos(new mjtNum[3]),
                                                                                                  target_quat(new mjtNum[4]),
                                                                                                  q0(q0),
                                                                                                  integration_dt(integration_dt),
                                                                                                  Kpos(Kpos),
                                                                                                  Kori(Kori),
                                                                                                  Kp(Kp),
                                                                                                  Kd(Kd),
                                                                                                  Kp_null(Kp_null),
                                                                                                  Kd_null(Kd_null),
                                                                                                  fjac(new mjtNum[6 * model->nv]),
                                                                                                  M_inv(new mjtNum[model->nv * model->nv]),
                                                                                                  tau(new mjtNum[7]),
                                                                                                  eye_mjt(new mjtNum[model->nv * model->nv]),
                                                                                                  fjac_eigen(fjac, 6, model->nv),
                                                                                                  M_inv_eigen(M_inv, model->nv, model->nv),
                                                                                                  eye(eye_mjt, model->nv, model->nv),
                                                                                                  Mx_eigen(6, 6)
        {
            // std::cout << "In constructor" << std::endl;
            mju_eye(eye_mjt, model->nv);
            twist_eigen = Eigen::VectorXd(6);
        }
        ~OpspaceController() {}

        void run_steps(mjData *d, int n_steps, mjtNum *target_pos, mjtNum *target_quat, mjtNum *jnt_ctrl, mjtNum *gripper_ctrl)
        {
            mju_copy3(this->target_pos, target_pos);
            mju_copy4(this->target_quat, target_quat);
            for (int i = 0; i < n_steps; ++i)
            {
                _run_step(d, i, jnt_ctrl, gripper_ctrl);
                mj_step(model, d);
            }
        }

    private:
        const mjModel *model;
        int site_id;
        int *dof_ids;
        int *actuator_ids;
        bool gravcomp;
        mjtNum *target_pos;
        mjtNum *target_quat;
        mjtNum *q0;
        double integration_dt;
        double Kpos;
        double Kori;
        mjtNum *Kp;
        mjtNum *Kd;
        mjtNum *Kp_null;
        mjtNum *Kd_null;
        mjtNum *fjac;
        mjtNum *M_inv;
        mjtNum *tau;
        mjtNum *eye_mjt;

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> fjac_eigen;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> M_inv_eigen;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eye;
        Eigen::VectorXd twist_eigen;
        Eigen::MatrixXd Mx_eigen;
        mjtNum site_quat[4];
        mjtNum site_quat_conj[4];
        mjtNum error_quat[4];

        void _run_step(mjData *d, int step, mjtNum *jnt_ctrl, mjtNum *gripper_ctrl)
        {
            Eigen::Map<Eigen::Vector3d> pos(target_pos);
            Eigen::Map<Eigen::Quaterniond> quat(target_quat);
            Eigen::Map<Eigen::VectorXd> tau_eigen(tau, 7);
            Eigen::Map<Eigen::VectorXd> Kp_eigen(Kp, 6);
            Eigen::Map<Eigen::VectorXd> Kd_eigen(Kd, 6);
            Eigen::Map<Eigen::VectorXd> Kp_null_eigen(Kp_null, 7);
            Eigen::Map<Eigen::VectorXd> Kd_null_eigen(Kd_null, 7);

            // Compute position error
            Eigen::Vector3d dx = pos - Eigen::Map<Eigen::Vector3d>(d->site_xpos + 3 * site_id);

            // Compute twist
            twist_eigen.head<3>() = Kpos * dx / integration_dt;
            mju_mat2Quat(site_quat, d->site_xmat + 9 * site_id);
            mju_negQuat(site_quat_conj, site_quat);
            mju_mulQuat(error_quat, target_quat, site_quat_conj);
            mju_quat2Vel(twist_eigen.data() + 3, error_quat, 1.0);
            twist_eigen.tail<3>() *= Kori / integration_dt;

            // Compute Jacobian
            mj_jacSite(model, d, fjac, fjac + 3 * model->nv, site_id);
            Eigen::MatrixXd jac = fjac_eigen(Eigen::all, Eigen::seq(dof_ids[0], dof_ids[6]));

            // Compute task-space inertia matrix
            mj_solveM(model, d, M_inv, eye_mjt, model->nv);
            Eigen::MatrixXd Mx_inv = jac * M_inv_eigen(Eigen::seq(dof_ids[0], dof_ids[6]), Eigen::seq(dof_ids[0], dof_ids[6])) * jac.transpose();

            if (std::abs(Mx_inv.determinant()) >= 1e-2)
            {
                Mx_eigen = Mx_inv.inverse();
            }
            else
            {
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(Mx_inv, Eigen::ComputeThinU | Eigen::ComputeThinV);
                Eigen::VectorXd singular_values = svd.singularValues();
                for (unsigned int i = 0; i < singular_values.size(); ++i)
                {
                    singular_values[i] = singular_values[i] / (singular_values[i] * singular_values[i] + 1e-2 * 1e-2);
                }
                Mx_eigen.noalias() = Eigen::MatrixXd(svd.matrixV() * singular_values.asDiagonal() * svd.matrixU().transpose());
            }

            // Compute generalized forces
            Eigen::VectorXd temp = Kp_eigen.cwiseProduct(twist_eigen) - Kd_eigen.cwiseProduct(jac * Eigen::Map<Eigen::VectorXd>(d->qvel + dof_ids[0], 7));
            tau_eigen = jac.transpose() * Mx_eigen * temp;

            // Add joint task in nullspace
            Eigen::MatrixXd Jbar = M_inv_eigen(Eigen::seq(dof_ids[0], dof_ids[6]), Eigen::seq(dof_ids[0], dof_ids[6])) * jac.transpose() * Mx_eigen;
            Eigen::VectorXd ddq = Kp_null_eigen.cwiseProduct(Eigen::Map<Eigen::VectorXd>(q0, 7) - Eigen::Map<Eigen::VectorXd>(d->qpos + dof_ids[0], 7)) - Kd_null_eigen.cwiseProduct(Eigen::Map<Eigen::VectorXd>(d->qvel + dof_ids[0], 7));
            tau_eigen += (Eigen::MatrixXd::Identity(7, 7) - jac.transpose() * Jbar.transpose()) * ddq;

            // Add gravity compensation
            if (gravcomp)
            {
                tau_eigen += Eigen::Map<Eigen::VectorXd>(d->qfrc_bias + dof_ids[0], 7);
            }
            
            // Clip tau values to actuator control range and assign to control data
            for (int i = 0; i < 7; ++i)
            {
                tau_eigen[i] = std::max(model->actuator_ctrlrange[actuator_ids[i] * 2], std::min(tau_eigen[i], model->actuator_ctrlrange[actuator_ids[i] * 2 + 1]));
                d->ctrl[actuator_ids[i]] = tau_eigen[i];
            }

            if (jnt_ctrl != nullptr)
            {
                memcpy(jnt_ctrl + step * 7, tau, 7 * sizeof(mjtNum));
            }

            // Assign gripper control
            if (gripper_ctrl != nullptr)
            {
                d->ctrl[actuator_ids[7]] = gripper_ctrl[step];
            }
            // std::cout << "tau["<<step<<"]: " << tau_eigen.transpose() << std::endl;

            return;
        }
    };

    PYBIND11_MODULE(opspace_ctrl, pymodule)
    {
        namespace py = ::pybind11;
        using PyCArray = py::array_t<mjtNum, py::array::c_style>;
        using PyIntCArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

        py::class_<OpspaceController>(pymodule, "OpspaceController")
            .def("mj_version", [](OpspaceController &self){ return mjVERSION_HEADER; })
            .def(py::init([](py::object m, int site_id, PyIntCArray dof_ids, PyIntCArray actuator_ids, bool gravcomp, PyCArray q0, double integration_dt,
                             double Kpos, double Kori, PyCArray Kp, PyCArray Kd, PyCArray Kp_null, PyCArray Kd_null)
                          {

            if (mjVERSION_HEADER != mj_version()) {
                throw std::runtime_error("MuJoCo library and header mismatch! mjVERSION_HEADER: " + std::to_string(mjVERSION_HEADER) + ", mj_version(): " + std::to_string(mj_version()));
            }

            std::uintptr_t m_raw = m.attr("_address").cast<std::uintptr_t>();
            const mjModel *model = reinterpret_cast<const mjModel*>(m_raw);

            auto dof_ids_buf = dof_ids.request();
            auto actuator_ids_buf = actuator_ids.request();
            auto q0_buf = q0.request();
            auto Kp_buf = Kp.request();
            auto Kd_buf = Kd.request();
            auto Kp_null_buf = Kp_null.request();
            auto Kd_null_buf = Kd_null.request();

            if (dof_ids_buf.ndim != 1 || actuator_ids_buf.ndim != 1 || q0_buf.ndim != 1 || Kp_buf.ndim != 1 || Kd_buf.ndim != 1 || Kp_null_buf.ndim != 1 || Kd_null_buf.ndim != 1) {
                throw std::runtime_error("Input arrays must be 1-dimensional");
            }

            if (q0_buf.size != 7) {
                throw std::runtime_error("q0 has wrong size: " + q0_buf.size);
            }

            int* dof_ids_ptr = static_cast<int*>(dof_ids_buf.ptr);
            int* actuator_ids_ptr = static_cast<int*>(actuator_ids_buf.ptr);
            mjtNum* q0_ptr = static_cast<mjtNum*>(q0_buf.ptr);
            mjtNum* Kp_ptr = static_cast<mjtNum*>(Kp_buf.ptr);
            mjtNum* Kd_ptr = static_cast<mjtNum*>(Kd_buf.ptr);
            mjtNum* Kp_null_ptr = static_cast<mjtNum*>(Kp_null_buf.ptr);
            mjtNum* Kd_null_ptr = static_cast<mjtNum*>(Kd_null_buf.ptr);

            return new OpspaceController(model, site_id, dof_ids_ptr, actuator_ids_ptr, gravcomp, q0_ptr, integration_dt, Kpos, Kori, Kp_ptr, Kd_ptr, Kp_null_ptr, Kd_null_ptr); }))
            .def("run_steps", [](OpspaceController &self, py::object d, int n_steps, PyCArray t_pos, PyCArray t_quat, std::optional<const PyCArray> jnt_ctrl, std::optional<const PyCArray> gripper_ctrl)
                 {
            std::uintptr_t d_raw = d.attr("_address").cast<std::uintptr_t>();
            mjData *data = reinterpret_cast<mjData*>(d_raw);

            auto pos_buf = t_pos.request();
            auto quat_buf = t_quat.request();

            mjtNum *jnt_ctrl_ptr = nullptr;
            if (jnt_ctrl.has_value())
            {
                py::buffer_info info = jnt_ctrl->request();
                if (info.size != 7 * n_steps)
                {
                    std::ostringstream msg;
                    msg << "Joint control array size should be " << 7 * n_steps << ", got " << info.size;
                    throw py::value_error(msg.str());
                }
                jnt_ctrl_ptr = static_cast<mjtNum*>(info.ptr);
            }

            mjtNum *gripper_ctrl_ptr = nullptr;
            if (gripper_ctrl.has_value())
            {
                py::buffer_info info = gripper_ctrl->request();
                if (info.size != 1 * n_steps)
                {
                    std::ostringstream msg;
                    msg << "Gripper control array size should be " << n_steps << ", got " << info.size;
                    throw py::value_error(msg.str());
                }
                gripper_ctrl_ptr = static_cast<mjtNum*>(info.ptr);
            }

            if (pos_buf.ndim != 1 || quat_buf.ndim != 1) {
                throw std::runtime_error("Input arrays must be 1-dimensional");
            }
            if (pos_buf.size != 3 || quat_buf.size != 4) {
                throw std::runtime_error("target_pos must have size 3 and target_quat size 4");
            }

            mjtNum* target_pos_ptr = static_cast<mjtNum*>(pos_buf.ptr);
            mjtNum* target_quat_ptr = static_cast<mjtNum*>(quat_buf.ptr);

            self.run_steps(data, n_steps, target_pos_ptr, target_quat_ptr, jnt_ctrl_ptr, gripper_ctrl_ptr); });
    }

} // end namespace mujoco::python
