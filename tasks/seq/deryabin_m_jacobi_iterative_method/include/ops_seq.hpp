#pragma once

#include "core/task/include/task.hpp"

#include <vector>

namespace deryabin_m_jacobi_iterative_method_seq {

class JacobiIterativeTaskSequential : public ppc::core::Task {
 public:
  explicit JacobiIterativeTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> 
  std::vector<double>
  std::vector<double>
};

}  // namespace deryabin_m_symbol_frequency_seq
