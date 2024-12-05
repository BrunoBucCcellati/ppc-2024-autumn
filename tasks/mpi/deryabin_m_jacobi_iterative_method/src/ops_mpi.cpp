#include "mpi/deryabin_m_jacobi_iterative_method/include/ops_mpi.hpp"

#include <thread>

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::pre_processing() {
  internal_order_test();
  input_matrix_ = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  input_right_vector_ = std::vector<double>(taskData->inputs_count[1]);
  auto* tmp_ptr2 = reinterpret_cast<double*>(taskData->inputs[1]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_matrix_[i] = tmp_ptr[i];
    if (i < taskData->inputs_count[1]) {
      input_right_vector_[i] = tmp_ptr2[i];
    }
  }
  output_x_vector_ = std::vector<double>(input_right_vector_.size());
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::validation() {
  internal_order_test();
  std::vector<double> matrix_ = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    matrix_[i] = tmp_ptr[i];
  }
  unsigned short i = 0;
  auto lambda = [&](double first, double second) { return (std::abs(first) + std::abs(second)); };
  while (i != sqrt(matrix_.size())) {
    if (i == 0) {
      if (std::abs(matrix_[0]) <=
          std::accumulate(matrix_.begin() + 1, matrix_.begin() + sqrt(matrix_.size()) - 1, 0, lambda)) {
        return false;
      }
    }
    if (i > 0 && i < sqrt(matrix_.size()) - 1) {
      if (std::abs(matrix_[i * (sqrt(matrix_.size()) + 1)]) <=
          std::accumulate(matrix_.begin() + i * sqrt(matrix_.size()),
                          matrix_.begin() + i * (sqrt(matrix_.size()) + 1) - 1, 0, lambda) +
              std::accumulate(matrix_.begin() + i * (sqrt(matrix_.size()) + 1) + 1,
                              matrix_.begin() + (i + 1) * sqrt(matrix_.size()) - 1, 0, lambda)) {
        return false;
      }
    }
    if (i == sqrt(matrix_.size()) - 1) {
      if (std::abs(matrix_[i * (sqrt(matrix_.size()) + 1)]) <=
          std::accumulate(matrix_.begin() + i * sqrt(matrix_.size()), matrix_.end() - 1, 0, lambda)) {
        return false;
      }
    }
    i++;
  }
  return taskData->outputs_count[0] == 1;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::run() {
  internal_order_test();
  unsigned short Nmax = 10000, num_of_iterations = 0;
  double epsilon = pow(10, -6), max_delta_x_i = 0;
  std::vector<double> x_old;
  do {
    x_old = output_x_vector_;
    unsigned short i = 0, j;
    double sum;
    while (i != sqrt(input_matrix_.size())) {
      j = 0;
      sum = 0;
      while (j != sqrt(input_matrix_.size())) {
        if (i != j) {
          sum += input_matrix_[i * sqrt(input_matrix_.size()) + j] * x_old[j];
        }
        j++;
      }
      output_x_vector_[i] =
          (input_right_vector_[i] - sum) * (1.0 / input_matrix_[i * (sqrt(input_matrix_.size()) + 1)]);
      if (std::abs(output_x_vector_[i] - x_old[i]) > max_delta_x_i) {
        max_delta_x_i = std::abs(output_x_vector_[i] - x_old[i]);
      }
      i++;
    }
    num_of_iterations++;
  } while (num_of_iterations < Nmax && max_delta_x_i > epsilon);
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0] = output_x_vector_;
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned short number_of_local_matrix_rows = 0;
  unsigned short ostatochnoe_chislo_strock = 0;
  unsigned short n = 0;
  if (world.rank() == 0) {
    number_of_local_matrix_rows = (int)(sqrt(taskData->inputs_count[0])) / world.size();
    ostatochnoe_chislo_strock = (int)(sqrt(taskData->inputs_count[0])) % world.size();
    input_matrix_ = std::vector<double>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    input_right_vector_ = std::vector<double>(taskData->inputs_count[1]);
    auto* tmp_ptr2 = reinterpret_cast<double*>(taskData->inputs[1]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_matrix_[i] = tmp_ptr[i];
      if (i < taskData->inputs_count[1]) {
        input_right_vector_[i] = tmp_ptr2[i];
      }
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, 
                 input_matrix_.data() + (proc - 1) * number_of_local_matrix_rows * (int)(sqrt(taskData->inputs_count[0])),
                 number_of_local_matrix_rows * (int)(sqrt(taskData->inputs_count[0])));
      world.send(proc, 0, input_right_vector_.data() + (proc - 1) * number_of_local_matrix_rows, 
                 number_of_local_matrix_rows);
    }
    n = (int)(sqrt(taskData->inputs_count[0]));
  }
  boost::mpi::broadcast(world, number_of_local_matrix_rows, 0);
  boost::mpi::broadcast(world, n, 0);
  local_input_matrix_part_ = std::vector<double>(number_of_local_matrix_rows * n);
  local_input_right_vector_part_ = std::vector<double>(number_of_local_matrix_rows);
  if (world.rank() == 0) {
    local_input_matrix_part_ = std::vector<double>(
        input_matrix_.end() - 
            (number_of_local_matrix_rows + ostatochnoe_chislo_strock) * sqrt(taskData->inputs_count[0]), 
        input_matrix_.end());
    local_input_right_vector_part_ = 
        std::vector<double>(input_right_vector_.end() - number_of_local_matrix_rows - ostatochnoe_chislo_strock,
                            input_right_vector_.end());
    output_x_vector_ = std::vector<double>(input_right_vector_.size());
  } else {
    world.recv(0, 0, local_input_matrix_part_.data(), number_of_local_matrix_rows * n);
    world.recv(0, 0, local_input_right_vector_part_.data(), number_of_local_matrix_rows);
  }
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::validation() {
  internal_order_test();
  unsigned short number_of_local_matrix_rows = 0;
  unsigned short ostatochnoe_chislo_strock = 0;
  std::vector<double> matrix_ = std::vector<double>(taskData->inputs_count[0]);
  if (world.rank() == 0) {
    number_of_local_matrix_rows = (int)(sqrt(taskData->inputs_count[0])) / world.size();
    ostatochnoe_chislo_strock = (int)(sqrt(taskData->inputs_count[0])) % world.size();
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      matrix_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, matrix_.data() + (proc - 1) * number_of_local_matrix_rows * (int)(sqrt(taskData->inputs_count[0])), 
                 number_of_local_matrix_rows * (int)(sqrt(taskData->inputs_count[0])));
    }
  }
  boost::mpi::broadcast(world, number_of_local_matrix_rows, 0);
  std::vector<double> loc_matrix_part_ = 
      std::vector<double>(number_of_local_matrix_rows * sqrt(taskData->inputs_count[0]));
  if (world.rank() == 0) {
    loc_matrix_part_ = std::vector<double>(
        matrix_.end() - (number_of_local_matrix_rows + ostatochnoe_chislo_strock) * sqrt(taskData->inputs_count[0]), 
        matrix_.end());
  } else {
    world.recv(0, 0, loc_matrix_part_.data(), number_of_local_matrix_rows * sqrt(taskData->inputs_count[0]));
  }
  unsigned short i = 0;
  auto lambda = [&](double first, double second) { return (std::abs(first) + std::abs(second)); };
  while (i != loc_matrix_part_.size() / sqrt(taskData->inputs_count[0])) {
    if (world.rank() == 1 && i == 0) {
      if (std::abs(loc_matrix_part_[0]) <=
          std::accumulate(loc_matrix_part_.begin() + 1, loc_matrix_part_.begin() + sqrt(taskData->inputs_count[0]) - 1, 
                          0, lambda)) {
        return false;
      }
    }
    if (world.rank() == 0) {
      if (i == number_of_local_matrix_rows + ostatochnoe_chislo_strock - 1) {
        if (std::abs(loc_matrix_part_[(i + 1) * sqrt(taskData->inputs_count[0]) - 1]) <=
            std::accumulate(loc_matrix_part_.begin() + i * sqrt(taskData->inputs_count[0]), loc_matrix_part_.end() - 1, 
                            0, lambda)) {
          return false;
        }
      } else {
        if (std::abs(loc_matrix_part_[(i + 1) * sqrt(taskData->inputs_count[0]) - 
                                      (number_of_local_matrix_rows + ostatochnoe_chislo_strock - i)]) <=
            std::accumulate(loc_matrix_part_.begin() + i * sqrt(taskData->inputs_count[0]), 
                            loc_matrix_part_.begin() + (i + 1) * sqrt(taskData->inputs_count[0]) - 
                                (number_of_local_matrix_rows + ostatochnoe_chislo_strock - i) - 1, 
                            0, lambda) + 
                std::accumulate(loc_matrix_part_.begin() + (i + 1) * sqrt(taskData->inputs_count[0]) - 
                                    (number_of_local_matrix_rows + ostatochnoe_chislo_strock - i) + 1, 
                                loc_matrix_part_.begin() + (i + 1) * sqrt(taskData->inputs_count[0]) - 1, 0, lambda)) {
          return false;
        }
      }
    } else {
      if (std::abs(loc_matrix_part_[i * sqrt(taskData->inputs_count[0]) + i + 
                                    (world.rank() - 1) * (number_of_local_matrix_rows)]) <=
          std::accumulate(loc_matrix_part_.begin() + i * sqrt(taskData->inputs_count[0]), 
                          loc_matrix_part_.begin() + i * sqrt(taskData->inputs_count[0]) + i + 
                              (world.rank() - 1) * (number_of_local_matrix_rows) - 1, 
                          0, lambda) + 
              std::accumulate(loc_matrix_part_.begin() + i * sqrt(taskData->inputs_count[0]) + i + 
                                  (world.rank() - 1) * (number_of_local_matrix_rows) + 1, 
                              loc_matrix_part_.begin() + (i + 1) * sqrt(taskData->inputs_count[0]) - 1, 0, lambda)) {
        return false;
      }
    }
    i++;
  }
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::run() {
  internal_order_test();
  double sendcounts;
  if (world.rank() == 0) {
    sendcounts = 1.0;
    //output_x_vector_[0] = 1;
      //boost::mpi::gatherv(world, output_x_vector_.data(), number_of_local_matrix_rows, 0);
      //boost::mpi::broadcast(world, output_x_vector_.data(), number_of_local_matrix_rows, 0);
  } else {
    sendcounts = 2.0;
    //boost::mpi::gather(world, sendcounts, 0);
    //output_x_vector_[1] = 1;
  }
  std::vector<double> v;
  boost::mpi::gather(world, sendcounts, v, 0);
  std::copy(v.begin(), v.end(), output_x_vector_.begin());
    //if (world.rank() == 0) {
      //boost::mpi::broadcast(world, output_x_vector_.data() + number_of_local_matrix_rows, number_of_local_matrix_rows + ostatochnoe_chislo_strock, 0);
    //} else {
      //boost::mpi::broadcast(world, output_x_vector_.data(), number_of_local_matrix_rows, 1); 
    //}
  //} while (num_of_iterations < Nmax);
  //boost::mpi::gatherv(world, output_x_vector_.data(), output_x_vector_, output_x_vector_, output_x_vector_, output_x_vector_, 0);
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0] = output_x_vector_;
  }
  return true;
} 
