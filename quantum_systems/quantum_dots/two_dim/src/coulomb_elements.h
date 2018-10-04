#ifndef COULOMB_FUNCTIONS_H
#define COULOMB_FUNCTIONS_H

double coulomb_ho(int ni, int mi, int nj, int mj, int nk, int mk, int nl,
        int ml);
double log_ratio_1(int int1, int int2, int int3, int int4);
double log_ratio_2(int G);
double log_factorial(int n);
double log_product_1(int n1, int m1, int n2, int m2, int n3, int m3,
        int n4, int m4);
double log_product_2(int n1, int m1, int n2, int m2, int n3, int m3,
        int n4, int m4, int j1, int j2, int j3, int j4);
double log_product_3(int l1, int l2, int l3, int l4, int g1, int g2,
        int g3, int g4);

#endif
