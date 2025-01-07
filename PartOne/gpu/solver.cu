/**
 * sudoku_cuda.cu
 *
 * Illustrative CUDA approach to tackling a "Sudoku + GCD" puzzle:
 *   - We store the board on the device.
 *   - For each possible excluded digit [0..9], we attempt to solve in parallel.
 *   - We keep track of the global best GCD and the corresponding filled board.
 *
 * This code is *not* production-level but demonstrates key ideas:
 *   - Splitting the puzzle across threads/blocks
 *   - Checking constraints on the device
 *   - Computing row integers and partial GCD
 *   - Storing the best (max) GCD solution in global memory
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include "solver.cuh"

// Thread workspace structure for parallel solving
struct ThreadWorkspace {
    int8_t board[TOTAL_CELLS];  // Local copy of the board for each thread
};

// ---------------------------------------------------------------------
// Device Utility Functions
// ---------------------------------------------------------------------

__device__ int64_t gcd_device(int64_t a, int64_t b) {
    while (b != 0) {
        int64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__device__ int64_t row_to_int_device(const int8_t* board, int row) {
    int start = row * BOARD_SIZE;
    int64_t val = 0;
    
    // Check for incomplete rows
    for (int j = 0; j < BOARD_SIZE; j++) {
        if (board[start + j] < 0) return -1;
    }
    
    // Convert to integer (handle leading zeros)
    for (int j = 0; j < BOARD_SIZE; j++) {
        val = val * 10 + board[start + j];
    }
    return val;
}

__device__ bool is_valid_move_device(
    const int8_t* board,
    int row, int col,
    int8_t val,
    int8_t excluded_digit
) {
    if (val == excluded_digit) return false;

    // Row check
    int row_start = row * BOARD_SIZE;
    for (int c = 0; c < BOARD_SIZE; c++) {
        if (board[row_start + c] == val) return false;
    }

    // Column check
    for (int r = 0; r < BOARD_SIZE; r++) {
        if (board[r * BOARD_SIZE + col] == val) return false;
    }

    // 3x3 box check
    int box_row = (row / 3) * 3;
    int box_col = (col / 3) * 3;
    for (int r = box_row; r < box_row + 3; r++) {
        for (int c = box_col; c < box_col + 3; c++) {
            if (board[r * BOARD_SIZE + c] == val) return false;
        }
    }
    return true;
}

__device__ int64_t compute_board_gcd_device(const int8_t* board) {
    int64_t gcd_val = 0;
    bool first_valid_row = true;

    for (int i = 0; i < BOARD_SIZE; i++) {
        int64_t row_val = row_to_int_device(board, i);
        if (row_val < 0) continue;

        if (first_valid_row) {
            gcd_val = row_val;
            first_valid_row = false;
        } else {
            gcd_val = gcd_device(gcd_val, row_val);
        }
    }
    return gcd_val;
}

// ---------------------------------------------------------------------
// Backtracking Solver
// ---------------------------------------------------------------------

__device__ void backtrack_solve(
    int8_t* best_board_out,
    int64_t* best_gcd_out,
    int8_t* board_work,
    int next_cell,
    int8_t excluded_digit,
    bool* solution_found,
    int64_t current_partial_gcd
) {
    if (next_cell >= TOTAL_CELLS) {
        int64_t final_gcd = compute_board_gcd_device(board_work);
        
        // Atomic update of best GCD if better
        int64_t old_gcd;
        do {
            old_gcd = *best_gcd_out;
            if (final_gcd <= old_gcd) return;
        } while (atomicCAS((unsigned long long*)best_gcd_out,
                          (unsigned long long)old_gcd,
                          (unsigned long long)final_gcd) != (unsigned long long)old_gcd);

        // Update best board
        for (int i = 0; i < TOTAL_CELLS; i++) {
            best_board_out[i] = board_work[i];
        }
        *solution_found = true;
        return;
    }

    // Early pruning: check if current partial GCD is worse than best
    if (current_partial_gcd > 0 && current_partial_gcd < *best_gcd_out) {
        return;
    }

    int row = next_cell / BOARD_SIZE;
    int col = next_cell % BOARD_SIZE;

    // Skip filled cells
    if (board_work[next_cell] >= 0) {
        backtrack_solve(best_board_out, best_gcd_out, board_work,
                       next_cell + 1, excluded_digit, solution_found,
                       current_partial_gcd);
        return;
    }

    // Try each possible digit
    for (int8_t val = 0; val < MAX_DIGITS; val++) {
        if (val == excluded_digit) continue;
        
        if (is_valid_move_device(board_work, row, col, val, excluded_digit)) {
            board_work[next_cell] = val;
            
            // Update partial GCD if we completed a row
            int64_t new_partial_gcd = current_partial_gcd;
            if (col == BOARD_SIZE - 1) {
                int64_t row_val = row_to_int_device(board_work, row);
                if (row_val > 0) {
                    new_partial_gcd = (current_partial_gcd == 0) ? 
                        row_val : gcd_device(current_partial_gcd, row_val);
                }
            }
            
            backtrack_solve(best_board_out, best_gcd_out, board_work,
                          next_cell + 1, excluded_digit, solution_found,
                          new_partial_gcd);
            
            board_work[next_cell] = -1;  // backtrack
        }
    }
}

// ---------------------------------------------------------------------
// CUDA Kernel
// ---------------------------------------------------------------------

__global__ void solve_sudoku_kernel(
    const int8_t* d_input_board,
    int8_t* d_best_board,
    int64_t* d_best_gcd,
    ThreadWorkspace* d_workspaces,
    bool* d_solution_found
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= MAX_DIGITS) return;

    int8_t excluded_digit = tid;

    // Check if excluded digit already exists in input
    for (int i = 0; i < TOTAL_CELLS; i++) {
        if (d_input_board[i] == excluded_digit) return;
    }

    // Initialize thread's workspace
    ThreadWorkspace* my_workspace = &d_workspaces[tid];
    for (int i = 0; i < TOTAL_CELLS; i++) {
        my_workspace->board[i] = d_input_board[i];
    }

    // Solve with this excluded digit
    backtrack_solve(
        d_best_board,
        d_best_gcd,
        my_workspace->board,
        0,
        excluded_digit,
        d_solution_found,
        0
    );
}

// ---------------------------------------------------------------------
// Host Interface
// ---------------------------------------------------------------------

cudaError_t SudokuSolverCUDA(
    const int8_t* h_board,
    int8_t* h_best_board,
    int64_t& h_best_gcd
) {
    cudaError_t cuda_status;
    int64_t zero = 0;
    bool false_val = false;
    int blocks = (MAX_DIGITS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Allocate device memory
    int8_t* d_input_board = nullptr;
    int8_t* d_best_board = nullptr;
    int64_t* d_best_gcd = nullptr;
    ThreadWorkspace* d_workspaces = nullptr;
    bool* d_solution_found = nullptr;

    cuda_status = cudaMalloc(&d_input_board, TOTAL_CELLS * sizeof(int8_t));
    if (cuda_status != cudaSuccess) return cuda_status;

    cuda_status = cudaMalloc(&d_best_board, TOTAL_CELLS * sizeof(int8_t));
    if (cuda_status != cudaSuccess) goto Error;

    cuda_status = cudaMalloc(&d_best_gcd, sizeof(int64_t));
    if (cuda_status != cudaSuccess) goto Error;

    cuda_status = cudaMalloc(&d_workspaces, MAX_DIGITS * sizeof(ThreadWorkspace));
    if (cuda_status != cudaSuccess) goto Error;

    cuda_status = cudaMalloc(&d_solution_found, sizeof(bool));
    if (cuda_status != cudaSuccess) goto Error;

    // Copy input data to device
    cuda_status = cudaMemcpy(d_input_board, h_board,
                            TOTAL_CELLS * sizeof(int8_t),
                            cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) goto Error;

    // Initialize device variables
    cudaMemcpy(d_best_gcd, &zero, sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_solution_found, &false_val, sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel
    solve_sudoku_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_input_board,
        d_best_board,
        d_best_gcd,
        d_workspaces,
        d_solution_found
    );

    // Check for kernel errors
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) goto Error;

    // Copy results back to host
    cuda_status = cudaMemcpy(h_best_board, d_best_board,
                            TOTAL_CELLS * sizeof(int8_t),
                            cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) goto Error;

    cuda_status = cudaMemcpy(&h_best_gcd, d_best_gcd,
                            sizeof(int64_t),
                            cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) goto Error;

Error:
    cudaFree(d_input_board);
    cudaFree(d_best_board);
    cudaFree(d_best_gcd);
    cudaFree(d_workspaces);
    cudaFree(d_solution_found);

    return cuda_status;
}

