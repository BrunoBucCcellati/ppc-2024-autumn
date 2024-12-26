#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

#include <thread>

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::pre_processing() {
  internal_order_test();
  input_matrix_A = std::vector<double>(taskData->inputs_count[0]);
  input_matrix_B = std::vector<double>(taskData->inputs_count[1]);
  auto* tmp_ptr_A = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* tmp_ptr_B = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(tmp_ptr_A, tmp_ptr_A + taskData->inputs_count[0], input_matrix_A.begin());
  std::copy(tmp_ptr_B, tmp_ptr_B + taskData->inputs_count[1], input_matrix_B.begin());
  output_matrix_C = std::vector<double>(input_matrix_A.size());
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->inputs_count[1] &&
         taskData->inputs_count[1] == pow((unsigned short)sqrt(taskData->inputs_count[0]), 2) &&
         taskData->outputs_count[0] == 1;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::run() {
  internal_order_test();
  unsigned short i = 0;
  unsigned short j;
  unsigned short count;
  auto dimension = (unsigned short)sqrt(input_matrix_A.size());
  while (i != dimension) {
    j = 0;
    while (j != dimension) {
      count = 0;
      while (count != dimension) {
        output_matrix_C[i * dimension + j] +=
            input_matrix_A[i * dimension + count] * input_matrix_B[count * dimension + j];
        count++;
      }
      j++;
    }
    i++;
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0] = output_matrix_C;
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_matrix_A = std::vector<double>(taskData->inputs_count[0]);
    input_matrix_B = std::vector<double>(taskData->inputs_count[1]);
    auto* tmp_ptr_A = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* tmp_ptr_B = reinterpret_cast<double*>(taskData->inputs[1]);
    std::copy(tmp_ptr_A, tmp_ptr_A + taskData->inputs_count[0], input_matrix_A.begin());
    std::copy(tmp_ptr_B, tmp_ptr_B + taskData->inputs_count[1], input_matrix_B.begin());
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == taskData->inputs_count[1] &&
           taskData->inputs_count[1] == pow((unsigned short)sqrt(taskData->inputs_count[0]), 2) &&
           taskData->outputs_count[0] == 1;
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::run() {
  internal_order_test();
  unsigned short i = 0;
  unsigned short j;
  unsigned short k;
  auto dimension = 0;
  unsigned short block_dimension = 0;
  unsigned short block_rows_columns = 0;
  if (world.rank() == 0) {
    dimension = (unsigned short)sqrt(input_matrix_A.size());
    output_matrix_C = std::vector<double>(dimension * dimension);
    block_rows_columns = (unsigned short)sqrt(world.size());
    block_dimension = dimension / block_rows_columns;
    //if (world.size() == 1 || world.size() != pow(block_rows_columns, 2) || dimension % block_rows_columns != 0) {
      //while (i != dimension) {
        //j = 0;
        //while (j != dimension) {
          //k = 0;
          //while (k != dimension) {
            //output_matrix_C[i * dimension + j] += input_matrix_A[i * dimension + k] * input_matrix_B[k * dimension + j];
            //k++;
          //}
          //j++;
        //}
        //i++;
      //}
      //return true;
    //}
  }
  boost::mpi::broadcast(world, dimension, 0);
  boost::mpi::broadcast(world, block_rows_columns, 0);
  boost::mpi::broadcast(world, block_dimension, 0);
  output_matrix_C = std::vector<double>(dimension * dimension);
  local_input_matrix_A = std::vector<double>(block_dimension * block_dimension);
  local_input_matrix_B = std::vector<double>(block_dimension * block_dimension);
  local_output_matrix_C = std::vector<double>(block_dimension * block_dimension);
  if (world.rank() == 0) {
    k = 0;
    while (k != block_dimension) {
      std::copy(input_matrix_A.data() + k * dimension, input_matrix_A.data() + k * dimension + block_dimension,
                local_input_matrix_A.begin() + k * block_dimension);
      std::copy(input_matrix_B.data() + k * dimension, input_matrix_B.data() + k * dimension + block_dimension,
                local_input_matrix_B.begin() + k * block_dimension);
      k++;
    }
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0] = output_matrix_C;
  }
  return true;
}
