#include <stdio.h>
#include <math.h> // CUDA IGNORE

#define MM 0.001f
#define DIM 3
#define PI 3.1415927f
#define X 0
#define Y 1
#define Z 2
#define Proton_Charge_Quantity 1.6021766208e-19f
#define Proton_Static_MassKg 1.672621898e-27f
#define Proton_Static_MassMeV 938.2720813f
#define Light_Speed 299792458.0f
#define RUN_STEP 0.001f

// 简单向量操作
void vct_cross(const float *a, const float *b, float *ret);

void vct_add_local(float *a_local, const float *b);

void vct_add(const float *a, const float *b, float *ret);

void vct_sub(const float *a, const float *b, float *ret);

void vct_dot_a_v(float a, float *v);

void vct_copy(const float *src, float *des);

float vct_len(const float *v);

void vct_neg(float *v);

// 磁场计算
void dB(float *p0, float *p1, float *p, float *ret);

int main() {
    int i;
    float sin_table[360];
    for (i = 0; i < 360; i++) {
        sin_table[i] = sinf(((float)i) / 180.0f * PI);
    }

    printf("Hello, World!%f\n",sinf(1.0f));
    return 0;
}

void vct_cross(const float *a, const float *b, float *ret) {
    ret[X] = a[Y] * b[Z] - a[Z] * b[Y];
    ret[Y] = -a[X] * b[Z] + a[Z] * b[X];
    ret[Z] = a[X] * b[Y] - a[Y] * b[X];
}

void vct_add_local(float *a_local, const float *b) {
    a_local[X] += b[X];
    a_local[Y] += b[Y];
    a_local[Z] += b[Z];
}

void vct_add(const float *a, const float *b, float *ret) {
    ret[X] = a[X] + b[X];
    ret[Y] = a[Y] + b[Y];
    ret[Z] = a[Z] + b[Z];
}

void vct_dot_a_v(float a, float *v) {
    v[X] *= a;
    v[Y] *= a;
    v[Z] *= a;
}

void vct_copy(const float *src, float *des) {
    des[X] = src[X];
    des[Y] = src[Y];
    des[Z] = src[Z];
}

float vct_len(const float *v) {
    return sqrtf(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
}

void vct_neg(float *v) {
    v[X] = -v[X];
    v[Y] = -v[Y];
    v[Z] = -v[Z];
}

void vct_sub(const float *a, const float *b, float *ret) {
    ret[X] = a[X] - b[X];
    ret[Y] = a[Y] - b[Y];
    ret[Z] = a[Z] - b[Z];
}

// 注意，这里计算的不是电流元的磁场，还需要乘以 电流 和 μ0/4π (=1e-7)
//__device__
//__forceinline__
void dB(float *p0, float *p1, float *p, float *ret) {
    float p01[DIM];
    float r[DIM];
    float rr;

    vct_sub(p1, p0, p01); // p01 = p1 - p0

    vct_add(p0, p1, r); // r = p0 + p1

    vct_dot_a_v(0.5f, r); // r = (p0 + p1)/2

    vct_sub(p, r, r); // r = p - r

    rr = vct_len(r); // rr = len(r)

    vct_cross(p01, r, ret); // ret = p01 x r

    rr = 1.0f / rr / rr / rr; // changed

    vct_dot_a_v(rr, ret); // rr . (p01 x r)
}



