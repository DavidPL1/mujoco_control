#include <mujoco/mujoco.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>
#include <pybind11/stl.h>
#include <math.h>

#include <iostream>
#include <sstream>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <algorithm> // for std::max and std::min

namespace mujoco::python
{
    namespace py = ::pybind11;

    using PyFArray = py::array_t<mjtNum, py::array::f_style>;
    using PyIntFArray = py::array_t<int, py::array::f_style | py::array::forcecast>;
    using PyCArray = py::array_t<mjtNum, py::array::c_style>;
    using PyIntCArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

    namespace
    {

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

        template <typename T>
        std::vector<T> numpy_to_vector(py::array_t<T> arr, int arr_len, const std::string &name) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.size != arr_len) {
                throw std::runtime_error("Input array `"+name+"` must be 1-dimensional and have length " + std::to_string(arr_len) + ", got ndim " + std::to_string(buf.ndim) + " and len " + std::to_string(buf.size));
            }
            T *ptr = static_cast<T *>(buf.ptr);
            return std::vector<T>(ptr, ptr + arr_len);
        }
    } // end namespace anonymous

    class OpspaceController
    {
    public:
        OpspaceController(
            const mjModel *model,
            int site_id, PyIntFArray dof_ids, PyIntFArray actuator_ids, bool gravcomp, PyFArray q0, double integration_dt,
            double Kpos, double Kori, PyFArray Kp, PyFArray Kd, PyFArray Kp_null, PyFArray Kd_null) : model(model),
                                                                                                  site_id(site_id),
                                                                                                  gravcomp(gravcomp),
                                                                                                  integration_dt(integration_dt),
                                                                                                  Kpos(Kpos),
                                                                                                  Kori(Kori)
        {
            this->dof_ids = numpy_to_vector<int>(dof_ids, 7, "dof_ids");
            std::cout << "dof_ids: [" << this->dof_ids[0] << " " << this->dof_ids[1] << " " << this->dof_ids[2] << " " << this->dof_ids[3] << " " << this->dof_ids[4] << " " << this->dof_ids[5] << " " << this->dof_ids[6] << "]" << std::endl; 
            this->actuator_ids = numpy_to_vector<int>(actuator_ids, 8, "actuator_ids");
            std::cout << "actuator_ids: [" << this->actuator_ids[0] << " " << this->actuator_ids[1] << " " << this->actuator_ids[2] << " " << this->actuator_ids[3] << " " << this->actuator_ids[4] << " " << this->actuator_ids[5] << " " << this->actuator_ids[6] << " " << this->actuator_ids[7] << "]" << std::endl;
            this->q0 = numpy_to_vector<mjtNum>(q0, 7, "q0");
            std::cout << "q0: [" << this->q0[0] << " " << this->q0[1] << " " << this->q0[2] << " " << this->q0[3] << " " << this->q0[4] << " " << this->q0[5] << " " << this->q0[6] << "]" << std::endl;
            this->Kp = numpy_to_vector<mjtNum>(Kp, 6, "Kp");
            std::cout << "Kp: [" << this->Kp[0] << " " << this->Kp[1] << " " << this->Kp[2] << " " << this->Kp[3] << " " << this->Kp[4] << " " << this->Kp[5] << "]" << std::endl;
            this->Kd = numpy_to_vector<mjtNum>(Kd, 6, "Kd");
            std::cout << "Kd: [" << this->Kd[0] << " " << this->Kd[1] << " " << this->Kd[2] << " " << this->Kd[3] << " " << this->Kd[4] << " " << this->Kd[5] << "]" << std::endl;
            this->Kp_null = numpy_to_vector<mjtNum>(Kp_null, 8, "Kp_null");
            std::cout << "Kp_null: [" << this->Kp_null[0] << " " << this->Kp_null[1] << " " << this->Kp_null[2] << " " << this->Kp_null[3] << " " << this->Kp_null[4] << " " << this->Kp_null[5] << " " << this->Kp_null[6] << "]" << std::endl;
            this->Kd_null = numpy_to_vector<mjtNum>(Kd_null, 8, "Kd_null");
            std::cout << "Kd_null: [" << this->Kd_null[0] << " " << this->Kd_null[1] << " " << this->Kd_null[2] << " " << this->Kd_null[3] << " " << this->Kd_null[4] << " " << this->Kd_null[5] << " " << this->Kd_null[6] << "]" << std::endl;

            this->fjac.resize(model->nv * 6);
            this->M_inv.resize(model->nv * model->nv);
            this->eye_mjt.resize(model->nv * model->nv);
 
            mju_eye(eye_mjt.data(), model->nv);
        }
        ~OpspaceController() {}

        void run_steps(mjData *d, int n_steps, mjtNum *target_pos, mjtNum *target_quat, std::optional<const PyCArray> jnt_ctrl, mjtNum *gripper_ctrl)
        {

            mj_markStack(d);
            mjtNum *jnt_ctrl_ptr = mj_stackAllocNum(d, 7*n_steps);

            mju_copy3(this->target_pos, target_pos);
            mju_copy4(this->target_quat, target_quat);
            // std::cout << "[C++] target_pos: [" << this->target_pos[0] << " " << this->target_pos[1] << " " << this->target_pos[2] << "]; target_quat: [" << this->target_quat[0] << " " << this->target_quat[1] << " "<< this->target_quat[2] << " "<< this->target_quat[3] << "];" << std::endl;
            for (int i = 0; i < n_steps; ++i)
            {
                _run_step(d, i, jnt_ctrl_ptr, gripper_ctrl);
                mj_step(model, d);
            }

            if (jnt_ctrl.has_value())
            {
                auto buf = jnt_ctrl->request();
                if (buf.size != 7 * n_steps) {
                    throw std::runtime_error("jnt_ctrl array must be of size " + std::to_string(7*n_steps));
                }
                mju_copy(static_cast<mjtNum*>(jnt_ctrl->request().ptr), jnt_ctrl_ptr, 7 * n_steps);
            }
            mj_freeStack(d);
        }

    private:
        const mjModel *model;
        int site_id;
        std::vector<int> dof_ids;
        std::vector<int> actuator_ids;
        bool gravcomp;
        mjtNum target_pos[3];
        mjtNum target_quat[4];
        std::vector<mjtNum> q0;
        double integration_dt;
        double Kpos;
        double Kori;
        std::vector<mjtNum> Kp;
        std::vector<mjtNum> Kd;
        std::vector<mjtNum> Kp_null;
        std::vector<mjtNum> Kd_null;
        std::vector<mjtNum> fjac;
        std::vector<mjtNum> M_inv;
        mjtNum tau[7];
        std::vector<mjtNum> eye_mjt;

        Eigen::Vector<double, 6> twist_eigen;
        Eigen::Matrix<double, 6,6> Mx_eigen;
        mjtNum site_quat[4];
        mjtNum site_quat_conj[4];
        mjtNum error_quat[4];

        void _run_step(mjData *d, int step, mjtNum *jnt_ctrl, mjtNum *gripper_ctrl)
        {
            Eigen::Map<Eigen::Vector3d> pos(target_pos);
            Eigen::Map<Eigen::Quaterniond> quat(target_quat);
            Eigen::Map<Eigen::VectorXd> tau_eigen(tau, 7);
            Eigen::Map<Eigen::VectorXd> Kp_eigen(Kp.data(), 6);
            Eigen::Map<Eigen::VectorXd> Kd_eigen(Kd.data(), 6);
            Eigen::Map<Eigen::VectorXd> Kp_null_eigen(Kp_null.data(), 7);
            Eigen::Map<Eigen::VectorXd> Kd_null_eigen(Kd_null.data(), 7);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> fjac_eigen(fjac.data(), 6, model->nv);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> M_inv_eigen(M_inv.data(), model->nv, model->nv);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eye(eye_mjt.data(), model->nv, model->nv);

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
            mj_jacSite(model, d, fjac.data(), fjac.data() + 3 * model->nv, site_id);
            Eigen::MatrixXd jac = fjac_eigen(Eigen::all, Eigen::seq(dof_ids.data()[0], dof_ids.data()[6]));

            // Compute task-space inertia matrix
            mj_solveM(model, d, M_inv.data(), eye_mjt.data(), model->nv);
            Eigen::MatrixXd Mx_inv = jac * M_inv_eigen(Eigen::seq(dof_ids.data()[0], dof_ids.data()[6]), Eigen::seq(dof_ids.data()[0], dof_ids.data()[6])) * jac.transpose();

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
            Eigen::VectorXd temp = Kp_eigen.cwiseProduct(twist_eigen) - Kd_eigen.cwiseProduct(jac * Eigen::Map<Eigen::VectorXd>(d->qvel + dof_ids.data()[0], 7));
            tau_eigen = jac.transpose() * Mx_eigen * temp;

            // Add joint task in nullspace
            Eigen::MatrixXd Jbar = M_inv_eigen(Eigen::seq(dof_ids.data()[0], dof_ids.data()[6]), Eigen::seq(dof_ids.data()[0], dof_ids.data()[6])) * jac.transpose() * Mx_eigen;
            Eigen::VectorXd ddq = Kp_null_eigen.cwiseProduct(Eigen::Map<Eigen::VectorXd>(q0.data(), 7) - Eigen::Map<Eigen::VectorXd>(d->qpos + dof_ids.data()[0], 7)) - Kd_null_eigen.cwiseProduct(Eigen::Map<Eigen::VectorXd>(d->qvel + dof_ids.data()[0], 7));
            tau_eigen += (Eigen::MatrixXd::Identity(7, 7) - jac.transpose() * Jbar.transpose()) * ddq;

            // Add gravity compensation
            if (gravcomp)
            {
                tau_eigen += Eigen::Map<Eigen::VectorXd>(d->qfrc_bias + dof_ids.data()[0], 7);
            }
            
            // Clip tau values to actuator control range and assign to control data
            for (int i = 0; i < 7; ++i)
            {
                tau_eigen[i] = std::max(model->actuator_ctrlrange[actuator_ids[i] * 2], std::min(tau_eigen[i], model->actuator_ctrlrange[actuator_ids[i] * 2 + 1]));
                d->ctrl[actuator_ids[i]] = tau_eigen[i];
            }

            mju_copy(jnt_ctrl + step * 7, tau, 7);

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

        py::class_<OpspaceController, std::shared_ptr<OpspaceController>>(pymodule, "OpspaceController")
            .def("mj_version", [](OpspaceController &self){ return mjVERSION_HEADER; })
            .def(py::init([](py::object m, int site_id, PyIntFArray dof_ids, PyIntFArray actuator_ids, bool gravcomp, PyFArray q0, double integration_dt,
                             double Kpos, double Kori, PyFArray Kp, PyFArray Kd, PyFArray Kp_null, PyFArray Kd_null)
                          {

            if (mjVERSION_HEADER != mj_version()) {
                throw std::runtime_error("MuJoCo library and header mismatch! mjVERSION_HEADER: " + std::to_string(mjVERSION_HEADER) + ", mj_version(): " + std::to_string(mj_version()));
            }

            std::uintptr_t m_raw = m.attr("_address").cast<std::uintptr_t>();
            const mjModel *model = reinterpret_cast<const mjModel*>(m_raw);

            return std::make_shared<OpspaceController>(model, site_id, dof_ids, actuator_ids, gravcomp, q0, integration_dt, Kpos, Kori, Kp, Kd, Kp_null, Kd_null); }))
            .def("run_steps", [](OpspaceController &self, py::object d, int n_steps, PyCArray t_pos, PyCArray t_quat, std::optional<const PyCArray> jnt_ctrl, std::optional<const PyCArray> gripper_ctrl)
                 {
            std::uintptr_t d_raw = d.attr("_address").cast<std::uintptr_t>();
            mjData *data = reinterpret_cast<mjData*>(d_raw);

            auto pos_buf = t_pos.request();
            auto quat_buf = t_quat.request();

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

            self.run_steps(data, n_steps, target_pos_ptr, target_quat_ptr, jnt_ctrl, gripper_ctrl_ptr); });
    }

} // end namespace mujoco::python
