#include <Eigen/Dense>
#include <iostream>
#include <ctime>
#include <cmath>

using namespace Eigen;

void set(MatrixXf& A, int m, int n, int id, int digits)
{
    for(auto i = 0; i < m; i++)
        for(auto j = 0; j < n; j++)
            A(i,j) = id*std::pow(10,(2*digits)) + i*std::pow(10,digits) + j;
}

int main(int argc, char* argv[])
{
#ifdef __DEBUG__
    int m = 9, k = 9, n = 9, max = std::max(std::max(m,k),n);
    MatrixXf A = MatrixXf::Zero(m, k);
    MatrixXf B = MatrixXf::Zero(k, n);
    MatrixXf C = MatrixXf::Zero(m, n);

    set(A, m, k, 1, static_cast<int>(std::log10(max)) + 1);
    set(B, k, n, 2, static_cast<int>(std::log10(max)) + 1);

    C = A*B;

    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;

    std::cout << std::endl;

    for(auto i = 0; i < m; i++)
    {
        for(auto j = 0; j < n; j++)
        {
            float acc=0;
            for(auto kk = 0; kk < k; kk++)
            {
                acc += A(i,kk)*B(kk,j);
            }
            C(i,j) = acc;
            //std::cout << acc << " ";
        }
        //std::cout << std::endl;
    }

    std::cout << C << std::endl;
#else
    int sz = 128;
    int m = sz, k = sz, n = sz;
    int RUNS = 500;
    double time = 0;

    for(auto i = 0; i < RUNS; i++)
    {
        MatrixXf A = MatrixXf::Random(m,k);
        MatrixXf B = MatrixXf::Random(k,n);
        //set(A,m, k, 1);
        //set(B,k, n, 2);
        MatrixXf C = MatrixXf::Zero(m, n);

        std::clock_t start,end;
        start = std::clock();
        C = A*B;
        end = std::clock();

        time += 1000.0*(end-start) / CLOCKS_PER_SEC;
    }
    std::cout << time << std::endl;
#ifdef TEST_SCALAR
    start = std::clock();
    for(auto i = 0; i < m; i++)
    {
        for(auto j = 0; j < n; j++)
        {
            float acc=0;
            for(auto kk = 0; kk < k; kk++)
            {
                acc += A(i,kk)*B(kk,j);
            }
            C(i,j) = acc;
        }
    }
    end = std::clock();

    std::cout << 1000.0*(end-start) / CLOCKS_PER_SEC << std::endl;
#endif
#endif
    return 0;
}