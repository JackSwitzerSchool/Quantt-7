#ifndef SUDOKU_CUDA_CUH
#define SUDOKU_CUDA_CUH

#include <cstdint>
#include <cuda_runtime.h>

// Constants for GPU configuration
constexpr int BOARD_SIZE = 9;
constexpr int TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE;
constexpr int MAX_DIGITS = 10;  // 0-9

// RTX 3070 specifications
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int WARP_SIZE = 32;
constexpr int SM_COUNT = 46;     // RTX 3070 has 46 SMs

// Optimal thread configuration for our use case
constexpr int THREADS_PER_BLOCK = 256;  // Multiple of warp size (32)
constexpr int MAX_BLOCKS = SM_COUNT * 2;  // 2 blocks per SM for good occupancy

extern "C" {
    /**
     * Main solver function to find the Sudoku solution with maximum GCD
     * 
     * @param h_board Input board (flat array, -1 for empty cells)
     * @param h_bestBoard Output board with best solution
     * @param h_bestGCD Best GCD value found
     * @return cudaError_t Status of the CUDA operation
     */
    cudaError_t SudokuSolverCUDA(
        const int8_t* h_board,
        int8_t* h_bestBoard,
        int64_t& h_bestGCD
    );
}

#endif // SUDOKU_CUDA_CUH 