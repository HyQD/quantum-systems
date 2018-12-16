#include <cmath>
#include <cassert>

#include "coulomb_elements.h"

/* Implementation of two-body matrix elements for the two-dimensional quantum
 * dots.
 *
 * Anisimovas, Matulis. J. Pys.: Condens. Matter 10, 601 (1998)
 *
 * Note that Anisimovas uses the convention that <pq|u|rs> is in fact <pg|u|sr>
 * in the standard way. That is, the two latter indices are interchanged.
 * */

double coulomb_ho(int ni, int mi, int nj, int mj, int nl, int ml, int nk,
        int mk)
{
    double element = 0.0;

    if (mi + mj != mk + ml) {
        return 0.0;
    }

    for (int j1 = 0; j1 <= ni; ++j1) {
        for (int j2 = 0; j2 <= nj; ++j2) {
            for (int j3 = 0; j3 <= nk; ++j3) {
                for (int j4 = 0; j4 <= nl; ++j4) {

                    int g1 = int(
                        j1 + j4 + 0.5 * (std::abs(mi) + mi)
                        + 0.5 * (std::abs(ml) - ml)
                    );
                    int g2 = int(
                        j2 + j3 + 0.5 * (std::abs(mj) + mj)
                        + 0.5 * (std::abs(mk) - mk)
                    );
                    int g3 = int(
                        j3 + j2 + 0.5 * (std::abs(mk) + mk)
                        + 0.5*(std::abs(mj) - mj)
                    );
                    int g4 = int(
                        j4 + j1 + 0.5 * (std::abs(ml) + ml)
                        + 0.5*(std::abs(mi) - mi)
                    );

                    int G = g1 + g2 + g3 + g4;

                    double ratio_1 = log_ratio_1(j1, j2, j3, j4);
                    double prod_2 = log_product_2(
                        ni, mi, nj, mj, nk, mk, nl, ml, j1, j2, j3, j4
                    );

                    double ratio_2 = log_ratio_2(G);

                    double temp = 0.0;

                    for (int l1 = 0; l1 <= g1; ++l1) {
                        for (int l2 = 0; l2 <= g2; ++l2) {
                            for (int l3 = 0; l3 <= g3; ++l3) {
                                for (int l4 = 0; l4 <= g4; ++l4) {

                                    if(l1 + l2 != l3 + l4) {
                                        continue;
                                    }

                                    int L = l1 + l2 + l3 + l4;

                                    temp += (
                                        -2 * ((g2 + g3 - l2 - l3) & 0x1) + 1
                                    ) * std::exp(
                                            log_product_3(
                                                l1, l2, l3, l4, g1, g2, g3, g4
                                            )
                                            + std::lgamma(1.0 + 0.5*L)
                                            + std::lgamma(0.5*(G - L + 1.0))
                                    );
                                }
                            }
                        }
                    }
                    element += (
                        -2 * ((j1 + j2 + j3 + j4) & 0x1) + 1
                    ) * std::exp(ratio_1 + prod_2 + ratio_2) * temp;
                }
            }
        }
    }

    element *= log_product_1(ni, mi, nj, mj, nk, mk, nl, ml);

    return element;
}

double log_factorial(int n)
{
    assert(n > 0);

    double fac = 0.0;
    for (int a = 2; a < n+1; a++) {
        fac += std::log(a);
    }

    return fac;
}

double log_ratio_1(int int1, int int2, int int3, int int4)
{
    return -log_factorial(int1) - log_factorial(int2) - log_factorial(int3)
        - log_factorial(int4);
}

double log_ratio_2(int G)
{
    return -0.5 * (G + 1) * std::log(2);
}

double log_product_1(int n1, int m1, int n2, int m2, int n3, int m3,
        int n4, int m4)
{
    double prod = log_factorial(n1) + log_factorial(n2) + log_factorial(n3)
        + log_factorial(n4);

    int arg1 = n1 + std::abs(m1);
    int arg2 = n2 + std::abs(m2);
    int arg3 = n3 + std::abs(m3);
    int arg4 = n4 + std::abs(m4);

    prod -= log_factorial(arg1) + log_factorial(arg2) + log_factorial(arg3)
        + log_factorial(arg4);

    return std::exp(0.5 * prod);
}

double log_product_2(int n1, int m1, int n2, int m2, int n3, int m3,
        int n4, int m4, int j1, int j2, int j3, int j4)
{
    int arg1 = n1 + std::abs(m1);
    int arg2 = n2 + std::abs(m2);
    int arg3 = n3 + std::abs(m3);
    int arg4 = n4 + std::abs(m4);

    int narg1 = n1 - j1;
    int narg2 = n2 - j2;
    int narg3 = n3 - j3;
    int narg4 = n4 - j4;

    int jarg1 = j1 + std::abs(m1);
    int jarg2 = j2 + std::abs(m2);
    int jarg3 = j3 + std::abs(m3);
    int jarg4 = j4 + std::abs(m4);

    double prod = log_factorial(arg1) + log_factorial(arg2)
        + log_factorial(arg3) + log_factorial(arg4);
    prod -= log_factorial(narg1) + log_factorial(narg2) + log_factorial(narg3)
        + log_factorial(narg4);
    prod -= log_factorial(jarg1) + log_factorial(jarg2) + log_factorial(jarg3)
        + log_factorial(jarg4);

    return prod;
}

double log_product_3(int l1, int l2, int l3, int l4, int g1, int g2,
        int g3, int g4)
{
    int garg1 = g1 - l1;
    int garg2 = g2 - l2;
    int garg3 = g3 - l3;
    int garg4 = g4 - l4;

    double prod = log_factorial(g1) + log_factorial(g2) + log_factorial(g3)
        + log_factorial(g4);
    prod -= log_factorial(l1) + log_factorial(l2) + log_factorial(l3)
        + log_factorial(l4);
    prod -= log_factorial(garg1) + log_factorial(garg2) + log_factorial(garg3)
        + log_factorial(garg4);

    return prod;
}
