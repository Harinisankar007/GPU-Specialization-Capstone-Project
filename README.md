# CUDA-Based-GPU-vs-GPU-Rock-Paper-Scissors-Game.
## Project Overview
## Name : Harini S
## Reg no : 212224240049

This project demonstrates a simple GPU vs GPU Rock Paper Scissors Game using CUDA. Two GPU players automatically generate moves (Rock, Paper, or Scissors) using CUDA kernels. The generated moves are compared, and the winner for each round is displayed in the terminal.

The project demonstrates CUDA programming concepts such as:

CUDA kernel execution
GPU computation
Parallel processing
Device-to-host memory transfer
Multi-round game execution

The game automatically executes several rounds and displays the results after every round.

## Features
```
GPU vs GPU gameplay
CUDA kernel execution
Automatic move generation
Multiple round execution
Terminal-based output visualization
Winner determination after every round
Demonstration of parallel processing concepts
```

## Technologies Used
```
CUDA Toolkit
NVIDIA GPU
C++
Google Colab / CUDA Environment
CUDA Runtime API
```

## Project Structure
```
GPU-RPS-CUDA/
│
├── gpu_rps_game.cu
├── README.md
├── output.txt
├── execution_log.txt
└── demo_video_link.txt
```

## Program
```
%%writefile gpu_rps_game.cu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

// CUDA Kernel to generate moves
__global__ void generateMoves(int *moves, int seed)
{
    int idx = threadIdx.x;

    // Generate values between 0 and 2
    moves[idx] = (seed + idx * 17) % 3;
}

// Convert number to move name
const char* getMoveName(int move)
{
    if(move == 0)
        return "Rock";

    else if(move == 1)
        return "Paper";

    else
        return "Scissors";
}

// Decide winner
void findWinner(int p1, int p2)
{
    printf("GPU 1 selected: %s\n", getMoveName(p1));
    printf("GPU 2 selected: %s\n", getMoveName(p2));

    if(p1 == p2)
    {
        printf("Result: Draw\n");
    }

    else if((p1 == 0 && p2 == 2) ||
            (p1 == 1 && p2 == 0) ||
            (p1 == 2 && p2 == 1))
    {
        printf("Winner: GPU 1\n");
    }

    else
    {
        printf("Winner: GPU 2\n");
    }
}

int main()
{
    int *d_moves;
    int h_moves[2];

    // Allocate GPU memory
    cudaMalloc((void**)&d_moves, 2 * sizeof(int));

    srand(time(0));

    // Execute 5 rounds
    for(int round = 1; round <= 5; round++)
    {
        printf("\n========================\n");
        printf("Round %d\n", round);
        printf("========================\n");

        int seed = rand();

        // Launch kernel with 2 threads
        generateMoves<<<1,2>>>(d_moves, seed);

        // Copy results from GPU to CPU
        cudaMemcpy(h_moves,
                   d_moves,
                   2 * sizeof(int),
                   cudaMemcpyDeviceToHost);

        // Display result
        findWinner(h_moves[0], h_moves[1]);
    }

    // Free GPU memory
    cudaFree(d_moves);

    printf("\nGame Finished\n");

    return 0;
}
```
## Output
<img width="336" height="658" alt="image" src="https://github.com/user-attachments/assets/826e9b06-9cea-4619-a0f5-cce614f0f39f" />

## Result
The CUDA-based GPU vs GPU Rock Paper Scissors game successfully generated moves using CUDA kernels, executed multiple rounds, compared the moves of both GPU players, and displayed the winner for each round using GPU parallel processing.
