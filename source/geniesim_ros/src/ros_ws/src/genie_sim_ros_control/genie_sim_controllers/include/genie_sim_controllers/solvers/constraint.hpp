#pragma once

#include <memory>
#include <utility>
#include "genie_sim_controllers/solvers/linear_relative_function.hpp"
#include "genie_sim_controllers/solvers/relative_function.hpp"
#include "genie_sim_controllers/solvers/cones.hpp"

namespace solvers
{

class Constraint : public FunctionBase
{
public:
  Constraint(size_t input_dim, size_t output_dim)
  : FunctionBase(input_dim, output_dim, true, false) {}

  void compute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt)
  {
    FunctionBase::compute(input, std::move(output_opt), std::move(jacobian_opt), std::nullopt);
  }

  virtual void getBounds(Eigen::Ref<Eigen::VectorXd> lb, Eigen::Ref<Eigen::VectorXd> ub) const = 0;
  [[nodiscard]] virtual const ConeBase & cone() const = 0;

protected:
  virtual void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt) = 0;

  void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt,
    FunctionBase::HessianArrayTypeOpt hessian_opt) override
  {
    (void)hessian_opt;
    doCompute(input, std::move(output_opt), std::move(jacobian_opt));
  }
};

class EqualityConstraint : public Constraint
{
public:
  EqualityConstraint(size_t input_dim, size_t output_dim, Eigen::VectorXd value)
  : Constraint(input_dim, output_dim), value_(std::move(value))
  {
    assert(value_.size() == output_dim);
  }

  [[nodiscard]] const ConeBase & cone() const override
  {
    static ZeroCone cone;
    return cone;
  }

  [[nodiscard]] virtual const Eigen::Ref<const Eigen::VectorXd> value() const {return value_;}

  virtual void setValue(Eigen::VectorXd value)
  {
    assert(value.size() == OUTPUT_DIM_);
    value_ = std::move(value);
    if (on_value_change_) {
      on_value_change_(*this);
    }
  }

  void getBounds(Eigen::Ref<Eigen::VectorXd> lb, Eigen::Ref<Eigen::VectorXd> ub) const override
  {
    lb = value();
    ub = value();
  }

  void setOnValueChangeCallback(std::function<void(const EqualityConstraint & constraint)> callback)
  {
    on_value_change_ = std::move(callback);
  }

protected:
  EqualityConstraint(size_t input_dim, size_t output_dim)
  : Constraint(input_dim, output_dim), value_(output_dim) {}
  Eigen::VectorXd value_;
  std::function<void(const EqualityConstraint & constraint)> on_value_change_;
};

class InequalityConstraint : public Constraint
{
public:
  InequalityConstraint(
    size_t input_dim, size_t output_dim, Eigen::VectorXd lower_bound,
    Eigen::VectorXd upper_bound)
  : Constraint(input_dim, output_dim), lower_bound_(std::move(lower_bound)),
    upper_bound_(std::move(upper_bound)), no_lower_bound_(false)
  {
    assert(lower_bound_.size() == output_dim);
    assert(upper_bound_.size() == output_dim);
  }

  InequalityConstraint(size_t input_dim, size_t output_dim, Eigen::VectorXd upper_bound)
  : Constraint(input_dim, output_dim),
    lower_bound_(Eigen::VectorXd::Constant(output_dim, std::numeric_limits<double>::min())),
    upper_bound_(std::move(upper_bound)), no_lower_bound_(true)
  {
    assert(upper_bound_.size() == output_dim);
  }

  [[nodiscard]] const ConeBase & cone() const override
  {
    static NegativeOrthogonalCone cone;
    return cone;
  }

  [[nodiscard]] bool hasLowerBound() const {return !no_lower_bound_;}
  [[nodiscard]] virtual Eigen::Ref<const Eigen::VectorXd> lowerBound() const {return lower_bound_;}
  [[nodiscard]] virtual Eigen::Ref<const Eigen::VectorXd> upperBound() const {return upper_bound_;}

  virtual void setLowerBound(Eigen::VectorXd lower_bound)
  {
    lower_bound_ = std::move(lower_bound);
  }
  virtual void setUpperBound(Eigen::VectorXd upper_bound)
  {
    upper_bound_ = std::move(upper_bound);
  }

  void getBounds(Eigen::Ref<Eigen::VectorXd> lb, Eigen::Ref<Eigen::VectorXd> ub) const override
  {
    lb = lowerBound();
    ub = upperBound();
  }

protected:
  InequalityConstraint(size_t input_dim, size_t output_dim, bool no_lower_bound)
  : Constraint(input_dim, output_dim),
    lower_bound_(output_dim),
    upper_bound_(output_dim),
    no_lower_bound_(no_lower_bound) {}
  Eigen::VectorXd lower_bound_;
  Eigen::VectorXd upper_bound_;
  bool no_lower_bound_;
};

class BoundConstraint : public InequalityConstraint
{
public:
  BoundConstraint(size_t input_dim, Eigen::VectorXd lower_bound, Eigen::VectorXd upper_bound)
  : BoundConstraint(
      std::make_shared<IdentityFunction>(input_dim), std::move(lower_bound), std::move(
        upper_bound)) {}

  BoundConstraint(
    const std::shared_ptr<LinearFunction> & linear_function,
    Eigen::VectorXd lower_bound, Eigen::VectorXd upper_bound)
  : InequalityConstraint(linear_function->input_dim(), linear_function->output_dim(),
      std::move(lower_bound), std::move(upper_bound)),
    linear_function_(linear_function) {}

  [[nodiscard]] std::shared_ptr<FunctionBase> deep_copy() const override
  {
    return std::make_shared<BoundConstraint>(
      linear_function_->deep_copy_as<LinearFunction>(), lower_bound_, upper_bound_);
  }

protected:
  void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt) override
  {
    linear_function_->compute(input, output_opt, jacobian_opt, std::nullopt);
  }

  std::shared_ptr<LinearFunction> linear_function_;
};

}  // namespace solvers
