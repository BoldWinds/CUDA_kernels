#include "kernels.h"
#include <iostream>
#include <string>
#include <map>
#include <vector>

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <problem> [options]\n\n";
    std::cerr << "Problems:\n";
    std::cerr << "  gemm            - General Matrix-Matrix Multiplication\n";
    std::cerr << "  matrix_add      - Matrix Addition\n\n";
    std::cerr << "Common Options:\n";
    std::cerr << "  --version <int>   Kernel version to run (default: 0)\n";
    std::cerr << "  --validate        Enable validation against CPU result\n\n";
    std::cerr << "GEMM Options:\n";
    std::cerr << "  --m <int>         Matrix A/C height\n";
    std::cerr << "  --n <int>         Matrix B/C width\n";
    std::cerr << "  --k <int>         Matrix A width / B height\n\n";
    std::cerr << "MatrixAdd Options:\n";
    std::cerr << "  --m <int>         Matrix height\n";
    std::cerr << "  --n <int>         Matrix width\n";
}

// Helper to find and parse an argument
int get_arg_val(int argc, char** argv, const std::string& arg_name, int default_val) {
    for (int i = 2; i < argc - 1; ++i) {
        if (std::string(argv[i]) == arg_name) {
            return std::stoi(argv[i + 1]);
        }
    }
    return default_val;
}

bool has_arg(int argc, char** argv, const std::string& arg_name) {
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == arg_name) {
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string problem = argv[1];
    int version = get_arg_val(argc, argv, "--version", 0);
    bool validate = has_arg(argc, argv, "--validate");

    float time_ms = -1.0f;

    if (problem == "gemm") {
        GemmArgs args;
        args.m = get_arg_val(argc, argv, "--m", 1024);
        args.n = get_arg_val(argc, argv, "--n", 1024);
        args.k = get_arg_val(argc, argv, "--k", 1024);
        args.lda = args.k;
        args.ldb = args.n;
        args.ldc = args.n;
        
        std::cout << "Running GEMM v" << version << " (m=" << args.m << ", n=" << args.n << ", k=" << args.k << ")\n";
        time_ms = launch_test_gemm(version, args, validate);

    } else if (problem == "matrix_add") {
        MatrixAddArgs args;
        args.m = get_arg_val(argc, argv, "--m", 1024);
        args.n = get_arg_val(argc, argv, "--n", 1024);
        args.lda = args.n;
        args.ldb = args.n;
        args.ldc = args.n;

        std::cout << "Running MatrixAdd v" << version << " (m=" << args.m << ", n=" << args.n << ")\n";
        time_ms = launch_test_matrix_add(version, args, validate);

    } else {
        std::cerr << "Error: Unknown problem '" << problem << "'\n";
        print_usage(argv[0]);
        return 1;
    }

    if (time_ms >= 0) {
        std::cout << "Execution Time: " << time_ms << " ms" << std::endl;
    }

    return 0;
}