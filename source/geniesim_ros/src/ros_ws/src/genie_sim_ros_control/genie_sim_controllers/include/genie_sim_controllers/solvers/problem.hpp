#pragma once

#include <Eigen/Sparse>
#include <memory>
#include <random>
#include "genie_sim_controllers/solvers/constraint.hpp"
#include "genie_sim_controllers/solvers/relative_function.hpp"

namespace solvers
{

class Problem
{
public:
  virtual ~Problem() = default;
  explicit Problem(const size_t N)
  : N_(N), init_guess_(N) {init_guess_.setZero();}

  template<typename Func>
  bool addCostFunction(
    const std::string & name,
    const RelativeFunctionWrapper<Func> & cost_function) requires(
    std::is_base_of_v<ScalarFunction,
    Func>) {
    return addCostFunction(
      name,
      cost_function.template deep_copy_as<RelativeFunctionWrapper<Func>>());
  }

  template<typename Func, template<typename> typename FuncWrapper>
  bool addCostFunction(
    const std::string & name,
    std::unique_ptr<FuncWrapper<Func>> && cost_function) requires(
    std::is_base_of_v<ScalarFunction, Func>&&
    std::is_base_of_v<RelativeFunctionWrapper<Func>, FuncWrapper<Func>>) {
    return addCostFunction(
      name,
      std::shared_ptr<RelativeFunctionWrapper<Func>>(std::move(cost_function)));
  }

  template<typename Func, template<typename> typename FuncWrapper>
  bool addCostFunction(const FuncWrapper<Func> & cost_function) requires(
    std::is_base_of_v<ScalarFunction, Func>&&
    std::is_base_of_v<RelativeFunctionWrapper<Func>, FuncWrapper<Func>>) {
    return addCostFunction(
      generateRandomString(64),
      cost_function.template deep_copy_as<RelativeFunctionWrapper<Func>>());
  }

  template<typename Func>
  auto * getCostFunction(const std::string & name) const requires(
    std::is_base_of_v<ScalarFunction,
    Func>) {
    if (const auto it = cost_function_map_.find(name); it != cost_function_map_.end()) {
      return dynamic_cast<Func *>(it->second.get());
    }
    return static_cast<Func *>(nullptr);
  }

  template<typename Func, template<typename> typename FuncWrapper>
  bool addConstraint(const std::string & name, const FuncWrapper<Func> & constraint) requires(
    std::is_base_of_v<Constraint, Func>&&
    std::is_base_of_v<RelativeFunctionWrapper<Func>, FuncWrapper<Func>>) {
    return addConstraint(
      name,
      constraint.template deep_copy_as<RelativeFunctionWrapper<Func>>());
  }

  template<typename Func, template<typename> typename FuncWrapper>
  bool addConstraint(const FuncWrapper<Func> & constraint) requires(
    std::is_base_of_v<Constraint, Func>&&
    std::is_base_of_v<RelativeFunctionWrapper<Func>, FuncWrapper<Func>>) {
    return addConstraint(
      generateRandomString(64),
      constraint.template deep_copy_as<RelativeFunctionWrapper<Func>>());
  }

  [[nodiscard]] size_t cost_size() const {return cost_function_map_.size();}
  [[nodiscard]] size_t equality_constraint_size() const {return equality_constraint_map_.size();}
  [[nodiscard]] size_t inequality_constraint_size() const
  {
    return inequality_constraint_map_.size();
  }
  [[nodiscard]] size_t constraint_size() const
  {
    return equality_constraint_size() + inequality_constraint_size();
  }

  bool makeProblem();

  [[nodiscard]] size_t N() const {return N_;}
  [[nodiscard]] size_t equality_constraint_dim() const {return equality_constraint_dim_;}
  [[nodiscard]] size_t inequality_constraint_dim() const {return inequality_constraint_dim_;}
  [[nodiscard]] size_t M() const {return M_;}

  [[nodiscard]] const auto & getCostFunctions() const {return cost_functions_;}
  [[nodiscard]] const auto & getEqualityConstraints() const {return equality_constraints_;}
  [[nodiscard]] const auto & getInequalityConstraints() const {return inequality_constraints_;}

  struct CostBlock
  {
    explicit CostBlock(const size_t dim)
    : input(dim), gradient(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dim))),
      hessian(Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(dim),
          static_cast
          <Eigen::Index>(dim))) {}
    Eigen::VectorXd input;
    double value{0};
    Eigen::VectorXd gradient;
    Eigen::MatrixXd hessian;
  };

  struct ConstraintBlock
  {
    explicit ConstraintBlock(const size_t input_dim, const size_t output_dim)
    : input(input_dim), value(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(output_dim))),
      jacobian(Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(output_dim),
        static_cast<Eigen::Index>(input_dim))) {}
    Eigen::VectorXd input;
    Eigen::VectorXd value;
    Eigen::MatrixXd jacobian;
  };

  class Context
  {
public:
    friend Problem;
    Context() = default;
    Context(const Context & other) = default;
    Context(Context && other) = default;
    [[nodiscard]] const std::vector<CostBlock> & getCostBlocks() const {return cost_blocks_;}
    [[nodiscard]] const std::vector<ConstraintBlock> & getEqualityBlocks() const
    {
      return equality_blocks_;
    }
    [[nodiscard]] const std::vector<ConstraintBlock> & getInequalityBlocks() const
    {
      return inequality_blocks_;
    }
    std::vector<CostBlock> & getCostBlocks() {return cost_blocks_;}
    std::vector<ConstraintBlock> & getEqualityBlocks() {return equality_blocks_;}
    std::vector<ConstraintBlock> & getInequalityBlocks() {return inequality_blocks_;}
    [[nodiscard]] size_t N() const {return N_;}
    [[nodiscard]] size_t M() const {return M_;}

private:
    size_t N_{0};
    size_t M_{0};
    std::vector<CostBlock> cost_blocks_;
    std::vector<ConstraintBlock> equality_blocks_;
    std::vector<ConstraintBlock> inequality_blocks_;
  };

  [[nodiscard]] const Eigen::VectorXd & init_guess() const {return init_guess_;}
  [[nodiscard]] Eigen::VectorXd & init_guess() {return init_guess_;}
  [[nodiscard]] std::unique_ptr<Context> createContext() const;

protected:
  static std::string generateRandomString(const size_t length)
  {
    const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, chars.size() - 1);
    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
      result += chars[dis(gen)];
    }
    return result;
  }

  template<typename Func, template<typename> typename FuncWrapper>
  bool addCostFunction(
    const std::string & name,
    const std::shared_ptr<FuncWrapper<Func>> & cost_function) requires(
    std::is_base_of_v<ScalarFunction, Func>&&
    std::is_base_of_v<RelativeFunctionWrapper<Func>, FuncWrapper<Func>>) {
    if (!checkFunction(*cost_function)) {return false;}
    return addCostFunctionImpl(name, cost_function);
  }

  template<typename Func, template<typename> typename FuncWrapper>
  bool addConstraint(
    const std::string & name,
    const std::shared_ptr<FuncWrapper<Func>> & constraint) requires(
    std::is_base_of_v<Constraint, Func>&&
    std::is_base_of_v<RelativeFunctionWrapper<Func>, FuncWrapper<Func>>) {
    if constexpr (std::is_base_of_v<EqualityConstraint, Func>) {
      return addEqualityConstraint(name, constraint);
    } else if constexpr (std::is_base_of_v<InequalityConstraint, Func>) {
      return addInequalityConstraint(name, constraint);
    } else {
      return false;
    }
  }

  template<typename Func, template<typename> typename FuncWrapper>
  bool addEqualityConstraint(
    const std::string & name,
    const std::shared_ptr<FuncWrapper<Func>> & constraint) requires(
    std::is_base_of_v<EqualityConstraint, Func>) {
    if (!checkFunction(*constraint)) {return false;}
    return addEqualityConstraintImpl(name, constraint);
  }

  template<typename Func, template<typename> typename FuncWrapper>
  bool addInequalityConstraint(
    const std::string & name,
    const std::shared_ptr<FuncWrapper<Func>> & constraint) requires(
    std::is_base_of_v<InequalityConstraint, Func>) {
    if (!checkFunction(*constraint)) {return false;}
    return addInequalityConstraintImpl(name, constraint);
  }

  bool addCostFunctionImpl(
    const std::string & name,
    const std::shared_ptr<ScalarFunction> & cost_function);
  bool addEqualityConstraintImpl(
    const std::string & name,
    const std::shared_ptr<EqualityConstraint> & equality_constraint);
  bool addInequalityConstraintImpl(
    const std::string & name,
    const std::shared_ptr<InequalityConstraint> & inequality_constraint);

  [[nodiscard]] bool checkFunction(const RelativeData & function) const;

  virtual bool doMakeProblem() = 0;
  virtual bool postMakeProblem() {return true;}

  size_t N_;
  size_t M_{0};
  size_t equality_constraint_dim_ = 0;
  size_t inequality_constraint_dim_ = 0;

  std::vector<std::pair<std::string, std::shared_ptr<ScalarFunction>>> cost_functions_;
  std::vector<std::pair<std::string, std::shared_ptr<EqualityConstraint>>> equality_constraints_;
  std::vector<std::pair<std::string,
    std::shared_ptr<InequalityConstraint>>> inequality_constraints_;

  std::map<std::string, std::shared_ptr<ScalarFunction>> cost_function_map_;
  std::map<std::string, std::shared_ptr<EqualityConstraint>> equality_constraint_map_;
  std::map<std::string, std::shared_ptr<InequalityConstraint>> inequality_constraint_map_;

  Eigen::VectorXd init_guess_;
};

}  // namespace solvers
