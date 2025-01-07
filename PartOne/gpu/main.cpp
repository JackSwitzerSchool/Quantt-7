#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include "board.hpp"
#include "solver.cuh"

// Helper function to print solution details
void print_solution_details(const SudokuBoard& board, int64_t gcd) {
    std::cout << "\nSolution found!\n";
    std::cout << "GCD: " << gcd << "\n\n";
    board.print_board();
    
    // Print the middle row (answer to the puzzle)
    auto flat_board = board.get_flat_board();
    std::cout << "\nMiddle row (puzzle answer): ";
    for (int i = BOARD_SIZE * 4; i < BOARD_SIZE * 5; i++) {
        std::cout << static_cast<int>(flat_board[i]);
    }
    std::cout << std::endl;
}

// Helper function to handle CUDA errors
void handle_cuda_error(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error during " << operation << ": "
                  << cudaGetErrorString(status) << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize board
    SudokuBoard board;
    std::string filename = (argc > 1) ? argv[1] : "board.csv";
    
    if (!board.load_board(filename)) {
        std::cerr << "Failed to load board from " << filename << std::endl;
        return 1;
    }

    // Get flat representation for CUDA
    auto flat_board = board.get_flat_board();
    std::vector<int8_t> solution(BOARD_SIZE * BOARD_SIZE);
    int64_t best_gcd = 0;

    // Call CUDA solver
    std::cout << "Starting CUDA solver...\n";
    cudaError_t cuda_status = SudokuSolverCUDA(
        flat_board.data(),
        solution.data(),
        best_gcd
    );

    // Handle any CUDA errors
    handle_cuda_error(cuda_status, "solver execution");

    // Update board with solution
    if (best_gcd > 0) {
        board.set_from_flat_board(solution);
        
        // Calculate and print execution time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        );

        // Print solution details
        print_solution_details(board, best_gcd);
        
        std::cout << "\nSolution found in " << duration.count() 
                  << " milliseconds\n";
    } else {
        std::cout << "No valid solution found!\n";
        return 1;
    }

    return 0;
} 