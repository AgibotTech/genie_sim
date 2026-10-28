#pragma once

#include <utility>
#include "genie_sim_controllers/solvers/relative_function.hpp"

namespace solvers
{

class LinearFunction : public FunctionBase
{
public:
  explicit LinearFunction(size_t output_dim)
  : FunctionBase(0, output_dim, true, false) {}
  LinearFunction(size_t input, size_t output_dim)
  : FunctionBase(input, output_dim, true, false) {}
};

class IdentityFunction final : public LinearFunction
{
public:
  IdentityFunction(size_t input_dim, Eigen::VectorXd r)
  : LinearFunction(input_dim, input_dim), r_(std::move(r)) {}
  explicit IdentityFunction(size_t input_dim)
  : LinearFunction(input_dim, input_dim), r_(Eigen::VectorXd::Zero((int)input_dim)) {}
  ~IdentityFunction() override = default;

  void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt,
    FunctionBase::HessianArrayTypeOpt) override
  {
    if (output_opt) {
      output_opt.value() = input + r_;
    }
    if (jacobian_opt) {
      jacobian_opt.value() = Eigen::MatrixXd::Identity((int)output_dim(), (int)input_dim());
    }
  }

  [[nodiscard]] std::shared_ptr<FunctionBase> deep_copy() const override
  {
    return std::make_shared<IdentityFunction>(*this);
  }

  const Eigen::VectorXd & r() const {return r_;}
  void set_r(const Eigen::VectorXd & r) {r_ = r;}

protected:
  Eigen::VectorXd r_;
};

class FilterIdentityFunction final : public LinearFunction
{
public:
  FilterIdentityFunction(size_t input_dim, const std::vector<size_t> & local_indices)
  : LinearFunction(input_dim, local_indices.size()), local_indices_(local_indices),
    r_(Eigen::VectorXd::Zero(
        (int)local_indices.size())) {}

  FilterIdentityFunction(
    size_t input_dim, const std::vector<size_t> & local_indices,
    const Eigen::VectorXd & r)
  : LinearFunction(input_dim, r.size()), local_indices_(local_indices), r_(r) {}

  ~FilterIdentityFunction() override = default;

  void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt,
    FunctionBase::HessianArrayTypeOpt) override
  {
    if (output_opt) {
      output_opt.value() = Eigen::VectorXd::Zero((int)output_dim());
      for (size_t i = 0; i < local_indices_.size(); ++i) {
        output_opt.value()((int)i) = input((int)local_indices_[i]) + r_((int)i);
      }
    }
    if (jacobian_opt) {
      jacobian_opt.value() = Eigen::MatrixXd::Zero((int)output_dim(), (int)input_dim());
      for (size_t i = 0; i < local_indices_.size(); ++i) {
        jacobian_opt.value()((int)i, (int)local_indices_[i]) = 1;
      }
    }
  }

  [[nodiscard]] std::shared_ptr<FunctionBase> deep_copy() const override
  {
    return std::make_shared<FilterIdentityFunction>(*this);
  }

  const Eigen::VectorXd & r() const {return r_;}
  void set_r(const Eigen::VectorXd & r) {r_ = r;}

protected:
  Eigen::VectorXd r_;
  std::vector<size_t> local_indices_;
};

}  // namespace solvers
