#include <stdio.h>
#include <math.h>

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

// 默认 CCT 分段为 3 度，这是满足计算精度下最粗的分段
#define STEP_KSI 3

// 粒子运动步长，默认 1mm
#define STEP_RUN 0.001f

// 倾斜角几个，默认 4 个，即二级场、四极场、六极场、八级场，如果修个这个参数，需要修改方法 ksi_phi_fun 因为为了性能写死了
#define TILE_ANGLE_LENGTH 4

#define SIN_45 0.7071067811865476f
#define COS_45 0.7071067811865476f

// 机架移动
#define CCT345_1_MOVE_X 5.680273403004535f
#define CCT345_1_MOVE_Y 2.279413679269048f

// 每匝 CCT 需要的参数 9 个 起点 ksi0，匝弧度 phi0，k[0][1][2][3]，极角 a，ch_eta0，sh_eta0，电流current，起点 phi_start
#define NUMBER_OF_VARIABLES_PER_CCT 11

// 简单向量常量操作
__device__ __forceinline__ void vct_cross(float *a, float *b, float *ret);

__device__ __forceinline__ void vct_add_local(float *a_local, float *b);

__device__ __forceinline__ void vct_add(float *a, float *b, float *ret);

__device__ __forceinline__ void vct_sub(float *a, float *b, float *ret);

__device__ __forceinline__ void vct_dot_a_v(float a, float *v);

__device__ __forceinline__ void vct_dot_a_v_ret(float a, float *v, float *ret);

__device__ __forceinline__ void vct_copy(float *src, float *des);

__device__ __forceinline__ float vct_len(float *v);

__device__ __forceinline__ void vct_zero(float *v);

__device__ __forceinline__ void vct_print(float *v);

__device__ __forceinline__ float deg2rad(int deg); // 角度转弧度。本代码中，角度一定是整数。这个方法，以后可能要打表
__device__ __forceinline__ float sin_deg(int deg); // 三角函数，参数整数的角度。这个方法，以后可能要打表。--re. 2020年11月14日 打表意义不大
__device__ __forceinline__ float cos_deg(int deg); // 同上

// 磁场计算 注意，这里计算的不是电流元的磁场，还需要乘以 电流 和 μ0/4π (=1e-7)
__device__ void dB(float *p0, float *p1, float *p, float *ret);

// ksi phi 函数。phi0 即一匝线圈后，大半径转过的弧度。k_tilt_angles 是倾斜角系数 == cot(倾斜角[i])/(i+1)sinh(eta)
__device__ __forceinline__ float ksi_phi_fun(int ksi_deg, float phi_start, float phi0, float *k_tilt_angles);

// 计算 CCT 上 ksi_deg 处的点，存放在 p_ret 中。k_tilt_angles 的含义见 ksi_phi_fun，a 是极角。ch_eta0 = ch(eta0)，sh_eta0 = sh(eta0)
__device__ __forceinline__ void
point_cct(int ksi_deg, float phi_start, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0,
          float *p_ret);

// 计算一匝 CCT 线圈在 p 点产生的磁场，注意磁场还要再乘电流 和 μ0/4π (=1e-7)
// ksi_deg0 是计算的起点。phi0、k_tilt_angles、a、ch_eta0、sh_eta0 见 point_cct 函数，p 点为需要计算磁场的点，m_ret 是返回的磁场
__device__ void
dB_cct_wind(int ksi_deg0, float phi_start, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0,
            float *p,
            float *m_ret);

// 粒子走一步 m 磁场，p 位置，v 速度，rm 动质量，sp 速率。默认步长 STEP_RUN == 1mm
__device__  __forceinline__  void particle_run_step(float *m, float *p, float *v, float run_mass, float speed);

// 在 Java 的 CCT 建模中，我们移动的是 CCT，将 CCT 平移 / 旋转 到指定的位置，但是这么做开销很大
// 与其移动带有上万个点的 CCT 模型，不如移动只有 1 个点的粒子。 p 为绝对坐标点，pr 为相对于 cct345_1（后偏转段第一段 CCT）的点
// 因此此函数的使用方法为，首先已知绝对坐标下的粒子 p，利用此函数求相对点 pr，然后进行磁场计算，得到的磁场也仅仅是相对磁场，
// 再利用 cct345_1_absolute_m 把相对磁场转为绝对磁场
// 此函数中带有大量的魔数，如果修改了机架模型的长度 / 位置，必须做出调整
__device__ __forceinline__ void cct345_1_relative_point(float *p, float *pr);

// 函数意义见 cct345_1_relative_point
__device__ __forceinline__ void cct345_1_absolute_m(float *mr, float *m);


/***************** DEFINE **********************/

__device__ __forceinline__ void vct_cross(float *a, float *b, float *ret) {
    ret[X] = a[Y] * b[Z] - a[Z] * b[Y];
    ret[Y] = -a[X] * b[Z] + a[Z] * b[X];
    ret[Z] = a[X] * b[Y] - a[Y] * b[X];
}

__device__ __forceinline__ void vct_add_local(float *a_local, float *b) {
    a_local[X] += b[X];
    a_local[Y] += b[Y];
    a_local[Z] += b[Z];
}

__device__ __forceinline__ void vct_add(float *a, float *b, float *ret) {
    ret[X] = a[X] + b[X];
    ret[Y] = a[Y] + b[Y];
    ret[Z] = a[Z] + b[Z];
}

__device__ __forceinline__ void vct_dot_a_v(float a, float *v) {
    v[X] *= a;
    v[Y] *= a;
    v[Z] *= a;
}

__device__ __forceinline__ void vct_dot_a_v_ret(float a, float *v, float *ret) {
    ret[X] = v[X] * a;
    ret[Y] = v[Y] * a;
    ret[Z] = v[Z] * a;
}

__device__ __forceinline__ void vct_copy(float *src, float *des) {
    des[X] = src[X];
    des[Y] = src[Y];
    des[Z] = src[Z];
}

__device__ __forceinline__ float vct_len(float *v) {
    return sqrtf(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
}

__device__ __forceinline__ void vct_zero(float *v) {
    v[X] = 0.0f;
    v[Y] = 0.0f;
    v[Z] = 0.0f;
}

__device__ __forceinline__ void vct_print(float *v) {
    printf("%f, %f, %f\n", v[X], v[Y], v[Z]);
}

__device__ __forceinline__ void vct_sub(float *a, float *b, float *ret) {
    ret[X] = a[X] - b[X];
    ret[Y] = a[Y] - b[Y];
    ret[Z] = a[Z] - b[Z];
}

__device__ __forceinline__ float deg2rad(int deg) {
    return ((float) deg) * PI / 180.0f;
}

__device__ __forceinline__ float sin_deg(int deg) {
    return __sinf(deg2rad(deg));
}

__device__ __forceinline__ float cos_deg(int deg) {
    return __cosf(deg2rad(deg));
}

__device__ void dB(float *p0, float *p1, float *p, float *ret) {
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

__device__ __forceinline__ float ksi_phi_fun(int ksi_deg, float phi_start, float phi0, float *k_tilt_angles) {
    // k 数组长度是 TILE_ANGLE_LENGTH，默认 4
    float ksi_rad = deg2rad(ksi_deg);

    return phi0 / (2.0f * PI) * ksi_rad +
           k_tilt_angles[0] * sin_deg(ksi_deg) +
           k_tilt_angles[1] * sin_deg(2 * ksi_deg) +
           k_tilt_angles[2] * sin_deg(3 * ksi_deg) +
           k_tilt_angles[3] * sin_deg(4 * ksi_deg) +
           phi_start;
}

__device__ __forceinline__ void
point_cct(int ksi_deg, float phi_start, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0,
          float *p_ret) {
    float phi = ksi_phi_fun(ksi_deg, phi_start, phi0, k_tilt_angles);
    float temp = a / (ch_eta0 - cos_deg(ksi_deg));

    p_ret[X] = temp * sh_eta0 * cosf(phi); // 太惨了，这个地方不能打表
    p_ret[Y] = temp * sh_eta0 * sinf(phi); // 太惨了，这个地方不能打表
    p_ret[Z] = temp * sin_deg(ksi_deg);
}

__device__ void
dB_cct_wind(int ksi_deg0, float phi_start, float phi0, float *k_tilt_angles, float a, float ch_eta0,
            float sh_eta0, float *p, float *m_ret) {
    int end_ksi_deg = ksi_deg0 + 360;
    float pre_point[3];
    float cur_point[3];
    float delta_B[3];

    point_cct(ksi_deg0, phi_start, phi0, k_tilt_angles, a, ch_eta0, sh_eta0, pre_point); // 起点

    vct_zero(m_ret); // m = 0,0,0

    while (ksi_deg0 < end_ksi_deg) {
        ksi_deg0 += STEP_KSI;

        point_cct(ksi_deg0, phi_start, phi0, k_tilt_angles, a, ch_eta0, sh_eta0, cur_point); // 下一个点

        dB(pre_point, cur_point, p, delta_B); // 计算磁场

        vct_add_local(m_ret, delta_B);

        vct_copy(cur_point, pre_point); // pre = cur
    }
}

__device__ __forceinline__ void cct345_1_relative_point(float *p, float *pr) {
    float px = p[X];
    float py = p[Y];
    float pz = p[Z];

    // 这两个魔数来自下面向量的相反数
    //Vector3 moving = afterDl2.moveSelf(
    //                directDl2.rotateSelf(BaseUtils.Converter.angleToRadian(-90))
    //                        .changeLengthSelf(secondBend.trajectoryBigRPart2))
    //                .toVector3();
    px -= CCT345_1_MOVE_X;
    py -= CCT345_1_MOVE_Y;

    // 下面是旋转
    // float r_phi = deg2rad(-135);
    // float c = -0.70710678f; // cos(-135) = - cos45
    // float s = -0.70710678f; // sin(-135) = - sin45
    // p[X] = c * x0 - s * y0;
    // p[Y] = s * x0 + c * y0;

    pr[X] = (px - py) * -SIN_45;
    pr[Y] = (px + py) * -COS_45;

    // xz 对称
    pr[Y] *= -1.f;

    // 填上 Z
    pr[Z] = pz;
}

// 函数意义见 cct345_1_relative_point
__device__ __forceinline__ void cct345_1_absolute_m(float *mr, float *m) {
    float mrx = mr[X];
    float mry = mr[Y];
    float mrz = mr[Z];

    // z
    m[Z] = mrz;

    // 对称回去
    mry *= -1.0f;

    // 旋转
    m[X] = (mrx + mry) * -SIN_45;
    m[Y] = (mrx - mry) * SIN_45;

    // 莫名其妙需要全反，我不知道为什么
    vct_dot_a_v(-1.0f, m);

//m[Y] *= -1.f;
//
//        float r_phi = deg2rad(135);
//        float c = cosf(r_phi); // -sin45
//        float s = sinf(r_phi); // sin45
//
//        float x0 = m[X];
//        float y0 = m[Y];
//
//        m[X] = c * x0 - s * y0;
//        m[Y] = s * x0 + c * y0;
//
//        vct_dot_a_v(-1.0f, m);
}

// 粒子走一步 m 磁场，p 位置，v 速度，rm 动质量，sp 速率
__device__  __forceinline__  void particle_run_step(float *m, float *p, float *v, float run_mass, float speed) {
    float a[3]; // 加速度
    float t;    // 运动时间
    float d[3]; // 位置变化 速度变化

    // q v b
    vct_cross(v, m, a); // a = v*b

    vct_dot_a_v(Proton_Charge_Quantity / run_mass, a); // a = q v b / mass 加速度

    t = STEP_RUN / speed; // 运动时长

    vct_dot_a_v_ret(t, v, d); // d = t v 位置变化

    vct_add_local(p, d); // p+=d

    vct_dot_a_v_ret(t, a, d); // d = t a 速度变化

    vct_add_local(v, d); // v+=d
}

/************  TEST *******************/
// 计算整个二极CCT产生的磁场。详见 test_magnet_bicct345_parallel_fill_data 2020年11月15日 测试通过
__global__ void test_magnet_bicct345_parallel(float *data) {
    unsigned int tid = threadIdx.x;

    float p[3] = {0, 0, 0};
    float m_per_wind[3];

    __shared__ float m_total[3];
    if (tid == 0) {
        vct_zero(m_total);
    }
    __syncthreads();

    // 外层
    // Java -0.004067504914360193, 0.006087451294501636, 0.011784791740462989
    // ret -0.004068, 0.006087, 0.011785
//    if (tid > 127 && tid < 256) { // 两层 bi cct

    // 内层
    // Java 0.0031436355039083964, -0.00470478301086915, 0.00888627084434009
    // ret 0.003144, -0.004705, 0.008886
//    if (tid < 128) { // 两层 bi cct

    // Java -9.238694104517966E-4, 0.0013826682836324865, 0.020671062584803078
    // ret -0.000924, 0.001383, 0.020671
    if (tid < 256) {
        dB_cct_wind(
                *((int *) (data + tid * NUMBER_OF_VARIABLES_PER_CCT + 0)), // ksi_deg0
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 10), // start_phi
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 1), // phi0
                data + tid * NUMBER_OF_VARIABLES_PER_CCT + 2, // k
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 6), // a
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 7), // ch_eta0
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 8), // sh_eta0
                p, m_per_wind
        );

        vct_dot_a_v((*(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 9)) * 1e-7f, m_per_wind);

        atomicAdd(&m_total[X], m_per_wind[X]);
        atomicAdd(&m_total[Y], m_per_wind[Y]);
        atomicAdd(&m_total[Z], m_per_wind[Z]);
    }

    __syncthreads();

    if (tid == 0) {
        vct_print(m_total);
    }
}

// 计算整个 AG-CCT 产出的磁场
// 计算整个二极CCT产生的磁场。详见 test_magnet_bicct345_parallel_fill_data 2020年11月15日 测试通过
__global__ void test_magnet_agcct345_parallel(float *data) {
    unsigned int tid = threadIdx.x;

    // Java -0.008153130729662327, 0.12075963610910645, -2.128470062057155
    float p[3] = {0.95, 0.1, 0}; // -0.008154, 0.120739, -2.128648
    float m_per_wind[3];

    __shared__ float m_total[3];
    if (tid == 0) {
        vct_zero(m_total);
    }
    __syncthreads();


    if (tid < 498) {
        dB_cct_wind(
                *((int *) (data + tid * NUMBER_OF_VARIABLES_PER_CCT + 0)), // ksi_deg0
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 10), // start_phi
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 1), // phi0
                data + tid * NUMBER_OF_VARIABLES_PER_CCT + 2, // k
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 6), // a
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 7), // ch_eta0
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 8), // sh_eta0
                p, m_per_wind
        );

        vct_dot_a_v((*(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 9)) * 1e-7f, m_per_wind);

        atomicAdd(&m_total[X], m_per_wind[X]);
        atomicAdd(&m_total[Y], m_per_wind[Y]);
        atomicAdd(&m_total[Z], m_per_wind[Z]);
    }

    __syncthreads();

    if (tid == 0) {
        vct_print(m_total);
    }
}

// 配合 test_magnet_bicct345_parallel 使用，填充参数到 h_data 中
void test_magnet_bicct345_parallel_fill_data(float *h_data) {
    int i;

    int BICCT345_INNER_WIND_NUMBER = 128;
    int BICCT345_OUTER_WIND_NUMBER = 128;
    int AGCCT345_INNER_WIND_NUMBER_0 = 21;
    int AGCCT345_INNER_WIND_NUMBER_1 = 50;
    int AGCCT345_INNER_WIND_NUMBER_2 = 50;
    int AGCCT345_OUTER_WIND_NUMBER_0 = 21;
    int AGCCT345_OUTER_WIND_NUMBER_1 = 50;
    int AGCCT345_OUTER_WIND_NUMBER_2 = 50;

    // bicct
    float bigR = 0.95f;
    float dicct_innerSmallR = 83 * MM + 15 * MM * 2;
    float dicct_outerSmall = 83 * MM + 15 * MM * 3;
    float dicct_bendingAngle_deg = 67.5f;
    float dicct_bendingRadian = dicct_bendingAngle_deg / 180.0f * PI;
    float dicct_tiltAngles[] = {30, 80, 90, 90};
    int dicct_windingNumber = 128;
    float dicct_current = -9664;
    float dicct_phi0 = dicct_bendingRadian / (float) dicct_windingNumber;

    float a, eta0, ch_eta0, sh_eta0, k[TILE_ANGLE_LENGTH];
    int ksi_deg0;

    if (1/*内层 BICCT 便于折叠 2020年11月15日 通过*/) {
        a = sqrtf(bigR * bigR - dicct_innerSmallR * dicct_innerSmallR);
        eta0 = 0.5f * logf((bigR + a) / (bigR - a));
        ch_eta0 = coshf(eta0);
        sh_eta0 = sinhf(eta0);

        k[0] = (1.0f / tanf(dicct_tiltAngles[0] / 180.0f * PI)) / ((float) (0 + 1) * sh_eta0);
        k[1] = (1.0f / tanf(dicct_tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
        k[2] = 0.0;
        k[3] = 0.0;

        ksi_deg0 = 0;
        for (i = 0; i < BICCT345_INNER_WIND_NUMBER; i++) { // 0-127
            *((int *) &h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 0]) = ksi_deg0 + 360 * i; // 黑科技

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = dicct_phi0;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0;
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = dicct_current;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = 0.0; // start_phi
        }
    }

    if (1/*外层 BICCT 便于折叠 2020年11月15日 通过*/) {
        dicct_current *= -1; // 电流一定要改，因为让线总是 ksi 正方向，不想 Java 中可以负方向 0~(-endKsi)


        a = sqrtf(bigR * bigR - dicct_outerSmall * dicct_outerSmall);
        eta0 = 0.5f * logf((bigR + a) / (bigR - a));
        ch_eta0 = coshf(eta0);
        sh_eta0 = sinhf(eta0);


        // 以下 5 个参数（4个 k，1 个 phi0）取了相反数，因为要对 ksi-phi 函数做 Y 轴对称
        // Java 代码是 phiKsiFun1 = phiKsiFun1.yAxisSymmetry();
        k[0] = -(1.0f / tanf(dicct_tiltAngles[0] / 180.0f * PI)) / ((float) (0 + 1) * sh_eta0);
        k[1] = -(1.0f / tanf(dicct_tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
        k[2] = -0.0;
        k[3] = -0.0;

        dicct_phi0 *= -1;

        ksi_deg0 = -360 * dicct_windingNumber;
        for (; i < BICCT345_INNER_WIND_NUMBER + BICCT345_OUTER_WIND_NUMBER; i++) { // 128-255
            /**
             * 巨大的 bug
             * 不是 ksi_deg0 + 360 * i
             * 而是 ksi_deg0 + 360 * (i - BICCT345_INNER_WIND_NUMBER);
             *
             * 应为 i 不再是从 0 开始了
             */
            *((int *) &h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 0]) =
                    ksi_deg0 + 360 * (i - BICCT345_INNER_WIND_NUMBER);

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = dicct_phi0;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0;
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = dicct_current;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = 0.0; // start_phi
        }
    }

    // agcct
    float agcct_innerSmallR = 83 * MM + 15 * MM * 0;
    float agcct_outerSmall = 83 * MM + 15 * MM * 1;
    float agcct_bendingAngle_degs[3] = {11.716404, 27.93897, 27.844626};
    float agcct_bendingRadians[3] = {agcct_bendingAngle_degs[0] / 180.0f * PI, agcct_bendingAngle_degs[1] / 180.0f * PI,
                                     agcct_bendingAngle_degs[2] / 180.0f * PI};
    float agcct_tiltAngles[] = {90, 30, 90, 90};
    int agcct_windingNumbers[3] = {21, 50, 50};
    float agcct_current = -6000;
    float agcct_phi0s[3] = {
            agcct_bendingRadians[0] / (float) agcct_windingNumbers[0],
            agcct_bendingRadians[1] / (float) agcct_windingNumbers[1],
            agcct_bendingRadians[2] / (float) agcct_windingNumbers[2]
    };

    int agcct_endKsis_deg[3] = {agcct_windingNumbers[0] * 360, agcct_windingNumbers[1] * 360,
                                agcct_windingNumbers[2] * 360};

    if (1/*内层 AGCCT1 便于折叠 2020年11月15日 通过*/) {
        a = sqrtf(bigR * bigR - agcct_innerSmallR * agcct_innerSmallR);
        eta0 = 0.5f * logf((bigR + a) / (bigR - a));
        ch_eta0 = coshf(eta0);
        sh_eta0 = sinhf(eta0);


        k[0] = 0.0;
        k[1] = (1.0f / tanf(agcct_tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
        k[2] = 0.0;
        k[3] = 0.0;

        ksi_deg0 = 0;

        for (; i < BICCT345_INNER_WIND_NUMBER + BICCT345_OUTER_WIND_NUMBER +
                   AGCCT345_INNER_WIND_NUMBER_0; i++) { // 256~256+21
            *((int *) &h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 0]) =
                    ksi_deg0 + 360 * (i - BICCT345_INNER_WIND_NUMBER - BICCT345_OUTER_WIND_NUMBER);

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[0];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0;
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = 0.0; // start_phi
        }
    }

    if (1/*内层 AGCCT2 便于折叠 2020年11月15日 通过*/) {
        agcct_current *= -1; // 因为反向

        // 反向了，k不反是因为反了两次
        // 第一次：AGCCT 本身交替 BaseUtils.ArrayUtils.dot(tiltAngles, -1)
        // 第二次：因为本 CUDA 代码，只支持正向绕线。原 Java 代码是反向绕线的
        k[0] = 0.0;
        k[1] = (1.0f / tanf(agcct_tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
        k[2] = 0.0;
        k[3] = 0.0;

        agcct_phi0s[1] *= -1; // 反向

        ksi_deg0 = -360 * agcct_windingNumbers[1]; // 负数，总是从小到大

        for (; i < BICCT345_INNER_WIND_NUMBER + BICCT345_OUTER_WIND_NUMBER +
                   AGCCT345_INNER_WIND_NUMBER_0 + AGCCT345_INNER_WIND_NUMBER_1; i++) { // 277~327
            *((int *) &h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 0]) = ksi_deg0 + 360 *
                                                                                 (i - BICCT345_INNER_WIND_NUMBER -
                                                                                  BICCT345_OUTER_WIND_NUMBER -
                                                                                  AGCCT345_INNER_WIND_NUMBER_0);

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[1];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0;
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current;

            // agcct_bendingRadians[0] + agcct_phi0s[0]
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = agcct_bendingRadians[0] + agcct_phi0s[0]; // start_phi
        }

        agcct_current *= -1; // 反回去
        agcct_phi0s[1] *= -1; // 反回去
    }

    if (1/*内层 AGCCT3 便于折叠 2020年11月15日 通过*/) {
        // 不用反向
        k[0] = 0.0;
        k[1] = (1.0f / tanf(agcct_tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
        k[2] = 0.0;
        k[3] = 0.0;

        ksi_deg0 = 0;

        for (; i < BICCT345_INNER_WIND_NUMBER + BICCT345_OUTER_WIND_NUMBER +
                   AGCCT345_INNER_WIND_NUMBER_0 + AGCCT345_INNER_WIND_NUMBER_1 +
                   AGCCT345_INNER_WIND_NUMBER_2; i++) { // [327-377)
            *((int *) &h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 0]) = ksi_deg0 + 360 *
                                                                                 (i - BICCT345_INNER_WIND_NUMBER -
                                                                                  BICCT345_OUTER_WIND_NUMBER -
                                                                                  AGCCT345_INNER_WIND_NUMBER_0 -
                                                                                  AGCCT345_INNER_WIND_NUMBER_1);

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[2];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0;
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] =
                    agcct_bendingRadians[0] + agcct_phi0s[0] + agcct_bendingRadians[1] + agcct_phi0s[1]; // start_phi
        }
    }

    if (1/*外层 AGCCT1 便于折叠 2020年11月15日 通过*/) {
        a = sqrtf(bigR * bigR - agcct_outerSmall * agcct_outerSmall);
        eta0 = 0.5f * logf((bigR + a) / (bigR - a));
        ch_eta0 = coshf(eta0);
        sh_eta0 = sinhf(eta0);

        // 反 .yAxisSymmetry();
        k[0] = -0.0;
        k[1] = -(1.0f / tanf(agcct_tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
        k[2] = -0.0;
        k[3] = -0.0;

        agcct_phi0s[0] *= -1; // 反

        ksi_deg0 = -360 * agcct_windingNumbers[0];
        agcct_current *= -1; // 反

        for (; i < BICCT345_INNER_WIND_NUMBER + BICCT345_OUTER_WIND_NUMBER +
                   AGCCT345_INNER_WIND_NUMBER_0 + AGCCT345_INNER_WIND_NUMBER_1 +
                   AGCCT345_INNER_WIND_NUMBER_2 + AGCCT345_OUTER_WIND_NUMBER_0; i++) { // [327-377)
            *((int *) &h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 0]) = ksi_deg0 + 360 *
                                                                                 (i - BICCT345_INNER_WIND_NUMBER -
                                                                                  BICCT345_OUTER_WIND_NUMBER -
                                                                                  AGCCT345_INNER_WIND_NUMBER_0 -
                                                                                  AGCCT345_INNER_WIND_NUMBER_1 -
                                                                                  AGCCT345_INNER_WIND_NUMBER_2);

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[0];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0;
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = 0.0; // start_phi
        }

        agcct_phi0s[0] *= -1; // 反回去
        agcct_current *= -1; // 反回去
    }

    if (1/*外层 AGCCT2 便于折叠 2020年11月15日 通过*/) {

        // 反 BaseUtils.ArrayUtils.dot(tiltAngles, -1)
        k[0] = -0.0;
        k[1] = -(1.0f / tanf(agcct_tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
        k[2] = -0.0;
        k[3] = -0.0;

        ksi_deg0 = 0;

        for (; i < BICCT345_INNER_WIND_NUMBER + BICCT345_OUTER_WIND_NUMBER +
                   AGCCT345_INNER_WIND_NUMBER_0 + AGCCT345_INNER_WIND_NUMBER_1 +
                   AGCCT345_INNER_WIND_NUMBER_2 + AGCCT345_OUTER_WIND_NUMBER_0 +
                   AGCCT345_OUTER_WIND_NUMBER_1; i++) { // 398 - 398+50
            *((int *) &h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 0]) = ksi_deg0 + 360 *
                                                                                 (i - BICCT345_INNER_WIND_NUMBER -
                                                                                  BICCT345_OUTER_WIND_NUMBER -
                                                                                  AGCCT345_INNER_WIND_NUMBER_0 -
                                                                                  AGCCT345_INNER_WIND_NUMBER_1 -
                                                                                  AGCCT345_INNER_WIND_NUMBER_2 -
                                                                                  AGCCT345_OUTER_WIND_NUMBER_0);

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[1];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0;
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = agcct_bendingRadians[0] + agcct_phi0s[0]; // start_phi
        }
    }

    if (1/*外层 AGCCT3 便于折叠 2020年11月15日 通过*/) {

        // .yAxisSymmetry();
        k[0] = -0.0;
        k[1] = -(1.0f / tanf(agcct_tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
        k[2] = -0.0;
        k[3] = -0.0;

        agcct_current *= -1;//反
        agcct_phi0s[2] *= -1;//反

        ksi_deg0 = -360 * agcct_windingNumbers[2];

        for (; i < BICCT345_INNER_WIND_NUMBER + BICCT345_OUTER_WIND_NUMBER +
                   AGCCT345_INNER_WIND_NUMBER_0 + AGCCT345_INNER_WIND_NUMBER_1 +
                   AGCCT345_INNER_WIND_NUMBER_2 + AGCCT345_OUTER_WIND_NUMBER_0 +
                   AGCCT345_OUTER_WIND_NUMBER_1 + AGCCT345_OUTER_WIND_NUMBER_2; i++) { // 348 - 398
            *((int *) &h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 0]) = ksi_deg0 + 360 *
                                                                                 (i - BICCT345_INNER_WIND_NUMBER -
                                                                                  BICCT345_OUTER_WIND_NUMBER -
                                                                                  AGCCT345_INNER_WIND_NUMBER_0 -
                                                                                  AGCCT345_INNER_WIND_NUMBER_1 -
                                                                                  AGCCT345_INNER_WIND_NUMBER_2 -
                                                                                  AGCCT345_OUTER_WIND_NUMBER_0 -
                                                                                  AGCCT345_OUTER_WIND_NUMBER_1);

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[2];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2];
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3];

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0;
            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current;

            h_data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] =
                    agcct_bendingRadians[0] + agcct_phi0s[0] + agcct_bendingRadians[1] + agcct_phi0s[1]; // start_phi
        }

        agcct_current *= -1; //反回去
        agcct_phi0s[2] *= -1; //反回去
    }
}

int main() {
    unsigned int blockNumber = 1;
    unsigned int threadNumber = 1024; // 最大CCT匝数和。当前 (128*2+(21+50+50)*2)*2)


    float *h_data; // 内存
    float *d_data; // 显存
    unsigned int data_size = blockNumber * threadNumber * NUMBER_OF_VARIABLES_PER_CCT * sizeof(float);

    h_data = (float *) malloc(data_size);
    cudaMalloc((void **) &d_data, data_size);

    test_magnet_bicct345_parallel_fill_data(h_data);

    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

    test_magnet_agcct345_parallel<<<blockNumber, threadNumber >>>(d_data);

    cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost);

    printf("hello -- host\n");

    free(h_data);
    cudaFree(d_data);

    return 0;
}

