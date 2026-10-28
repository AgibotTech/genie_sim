#include "genie_sim_controllers/solvers/scalar_function.hpp"

namespace solvers
{

void ScalarFunction::doCompute(
  FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
  FunctionBase::JacobianTypeOpt jacobian_opt,
  FunctionBase::HessianArrayTypeOpt hessian_opt)
{
  std::optional<double *> scalar_output_opt = output_opt.has_value() ? std::optional<double *>(
    &output_opt.value()(0)) : std::nullopt;
  std::optional<Eigen::Ref<Eigen::VectorXd>> scalar_gradient_opt =
      jacobian_opt.has_value() ?
      std::optional<Eigen::Ref<Eigen::VectorXd>>(
    Eigen::Map<Eigen::VectorXd>(
      jacobian_opt.value().data(),
      jacobian_opt.value().cols())) :
      std::nullopt;
  std::optional<Eigen::Ref<Eigen::MatrixXd>> scalar_hessian_opt =
      hessian_opt.has_value() ?
      std::optional<Eigen::Ref<Eigen::MatrixXd>>(hessian_opt.value().front())
      :
      std::nullopt;

  doCompute(input, scalar_output_opt, scalar_gradient_opt, scalar_hessian_opt);
}

}  // namespace solvers
