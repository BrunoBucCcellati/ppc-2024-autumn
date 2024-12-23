// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/muhina_m_shell_sort/include/ops_seq.hpp"

namespace muhina_m_shell_sort_mpi {

std::vector<int> Get_Random_Vector(int sz, int min_value, int max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min_value + gen() % (max_value - min_value + 1);
  }
  return vec;
}
}  // namespace muhina_m_shell_sort_mpi

TEST(muhina_m_shell_sort, test_pipeline_run) {
  const int count = 2500000;
  const int min_val = 0;
  const int max_val = 100;
  std::vector<int> in = muhina_m_shell_sort_mpi::Get_Random_Vector(count, min_val, max_val);
  std::vector<int> out(in.size(), 0);
  std::vector<int> expected_result = in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto ShellSortSequential = std::make_shared<muhina_m_shell_sort_seq::ShellSortSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ShellSortSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  std::sort(expected_result.begin(), expected_result.end());
  ASSERT_EQ(out, expected_result);
}

TEST(muhina_m_shell_sort, test_task_run) {
  const int count = 2500000;
  const int min_val = 0;
  const int max_val = 100;
  std::vector<int> in = muhina_m_shell_sort_mpi::Get_Random_Vector(count, min_val, max_val);
  std::vector<int> out(in.size(), 0);
  std::vector<int> expected_result = in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto ShellSortSequential = std::make_shared<muhina_m_shell_sort_seq::ShellSortSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ShellSortSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::sort(expected_result.begin(), expected_result.end());
  ASSERT_EQ(out, expected_result);
}
