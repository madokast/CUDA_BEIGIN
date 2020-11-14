#include <stdio.h>

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

// 简单向量常量操作
__device__ __forceinline__ void vct_cross(float *a, float *b, float *ret);

__device__ __forceinline__ void vct_add_local(float *a_local, float *b);

__device__ __forceinline__ void vct_add(float *a, float *b, float *ret);

__device__ __forceinline__ void vct_sub(float *a, float *b, float *ret);

__device__ __forceinline__ void vct_dot_a_v(float a, float *v);

__device__ __forceinline__ void vct_dot_a_v_ret(float a, float *v, float *ret);

__device__ __forceinline__ void vct_copy(float *src, float *des);

__device__ __forceinline__ float vct_len(float *v);

__device__ __forceinline__ void vct_neg(float *v);

__device__ __forceinline__ void vct_zero(float *v);

__device__ __forceinline__ void vct_print(float *v);

__device__ __forceinline__ float deg2rad(int deg); // 角度转弧度。本代码中，角度一定是整数。这个方法，以后可能要打表
__device__ __forceinline__ float sin_deg(int deg); // 三角函数，参数整数的角度。这个方法，以后可能要打表。--re. 2020年11月14日 打表意义不大
__device__ __forceinline__ float cos_deg(int deg); // 同上

// 磁场计算 注意，这里计算的不是电流元的磁场，还需要乘以 电流 和 μ0/4π (=1e-7)
__device__ void dB(float *p0, float *p1, float *p, float *ret);

// ksi phi 函数。phi0 即一匝线圈后，大半径转过的弧度。k_tilt_angles 是倾斜角系数 == cot(倾斜角[i])/(i+1)sinh(eta)
__device__ __forceinline__ float ksi_phi_fun(int ksi_deg, float phi0, float *k_tilt_angles);

// 计算 CCT 上 ksi_deg 处的点，存放在 p_ret 中。k_tilt_angles 的含义见 ksi_phi_fun，a 是极角。ch_eta0 = ch(eta0)，sh_eta0 = sh(eta0)
__device__ __forceinline__ void
point_cct(int ksi_deg, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0, float *p_ret);

// 计算一匝 CCT 线圈在 p 点产生的磁场，注意磁场还要再乘电流 和 μ0/4π (=1e-7)
// ksi_deg0 是计算的起点。phi0、k_tilt_angles、a、ch_eta0、sh_eta0 见 point_cct 函数，p 点为需要计算磁场的点，m_ret 是返回的磁场
__device__ void
dB_cct_wind(int ksi_deg0, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0, float *p,
            float *m_ret);

// 计算一层 CCT 在 p 点产生的磁场。注意磁场还要再乘电流 和 μ0/4π (=1e-7)
// ksi_deg0 是计算的起点。wind_num 是匝数。
// phi0、k_tilt_angles、a、ch_eta0、sh_eta0 见 point_cct 函数，p 点为需要计算磁场的点，m_ret 是返回的磁场。
__device__ void
magnet_cct(int ksi_deg0, int wind_num, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0,
           float *p, float *m_ret);

// 粒子走一步 m 磁场，p 位置，v 速度，rm 动质量，sp 速率。默认步长 STEP_RUN == 1mm
__device__  __forceinline__  void particle_run_step(float *m, float *p, float *v, float run_mass, float speed);

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

__device__ __forceinline__ void vct_neg(float *v) {
    v[X] = -v[X];
    v[Y] = -v[Y];
    v[Z] = -v[Z];
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

__device__ __forceinline__ float ksi_phi_fun(int ksi_deg, float phi0, float *k_tilt_angles) {
    // k 数组长度是 TILE_ANGLE_LENGTH，默认 4
    float ksi_rad = deg2rad(ksi_deg);

    return phi0 / (2.0f * PI) * ksi_rad + k_tilt_angles[0] * sin_deg(ksi_deg) +
           k_tilt_angles[1] * sin_deg(2 * ksi_deg) +
           k_tilt_angles[2] * sin_deg(3 * ksi_deg) +
           k_tilt_angles[3] * sin_deg(4 * ksi_deg);
}

__device__ __forceinline__ void
point_cct(int ksi_deg, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0, float *p_ret) {
    float phi = ksi_phi_fun(ksi_deg, phi0, k_tilt_angles);
    float temp = a / (ch_eta0 - cos_deg(ksi_deg));

    p_ret[X] = temp * sh_eta0 * cosf(phi); // 太惨了，这个地方不能达表
    p_ret[Y] = temp * sh_eta0 * sinf(phi); // 太惨了，这个地方不能达表
    p_ret[Z] = temp * sin_deg(ksi_deg);
}

__device__ void
dB_cct_wind(int ksi_deg0, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0, float *p,
            float *m_ret) {
    int end_ksi_deg = ksi_deg0 + 360;
    float pre_point[3];
    float cur_point[3];
    float delta_B[3];

    point_cct(ksi_deg0, phi0, k_tilt_angles, a, ch_eta0, sh_eta0, pre_point); // 起点

    vct_zero(m_ret); // m = 0,0,0

    while (ksi_deg0 < end_ksi_deg) {
        ksi_deg0 += STEP_KSI;

        point_cct(ksi_deg0, phi0, k_tilt_angles, a, ch_eta0, sh_eta0, cur_point); // 下一个点

        dB(pre_point, cur_point, p, delta_B); // 计算磁场

        vct_add_local(m_ret, delta_B);

        vct_copy(cur_point, pre_point); // pre = cur
    }
}

__device__ void // 并行度低!! 仅仅用于测试
magnet_cct(int ksi_deg0, int wind_num, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0,
           float *p, float *m_ret) {
    int wi;
    float m_pre_wind[3];
    vct_zero(m_ret); // m = 0,0,0

    for (wi = 0; wi < wind_num; wi++) {
        dB_cct_wind(ksi_deg0, phi0, k_tilt_angles, a, ch_eta0, sh_eta0, p, m_pre_wind);
        ksi_deg0 += 360;
        vct_add_local(m_ret, m_pre_wind);
    }
}

// 并发执行。这个方法和非同步的 magnet_cct 最大的不同是，返回值 m_ret 是真实磁场，不需要 current * 1e-7f，因此需要传入 CCT 电流 current
// 注意 m_ret 必须是 __shared__，其他参数意义见 magnet_cct。m_ret 不必置零，在此方法内完成
// 另外此方法本身就是同步方法，因为最后一行代码是 __syncthreads();，所以调用后无需执行额外的
__device__ void
magnet_cct_parallel(int ksi_deg0, int wind_num, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0,
                    float current, float *p, /*__shared__*/ float *m_ret) {
    int wi = (int) threadIdx.x; // 块内计算，每块计算一个 CCT，wind_num ≤ threadIdx

    if (wi == 0) {
        vct_zero(m_ret);
    }

    // 对共享内存写其他线程看得到
    __syncthreads();

    float m_part[3];

    if (wi < wind_num) {
        dB_cct_wind(ksi_deg0 + 360 * wi, phi0, k_tilt_angles, a, ch_eta0, sh_eta0, p, m_part);
        vct_dot_a_v(current * 1e-7f, m_part);
    }

    atomicAdd(&m_ret[X], m_part[X]);
    atomicAdd(&m_ret[Y], m_part[Y]);
    atomicAdd(&m_ret[Z], m_part[Z]);

    // 对共享内存写所有线程完成
    __syncthreads();
}

// 在 Java 的 CCT 建模中，我们移动的是 CCT，将 CCT 平移 / 旋转 到指定的位置，但是这么做开销很大
// 与其移动带有上万个点的 CCT 模型，不如移动只有 1 个点的粒子。 p 为绝对坐标点，pr 为相对于 cct345_1（后偏转段第一段 CCT）的点
// 因此此函数的使用方法为，首先已知绝对坐标下的粒子 p，利用此函数求相对点 pr，然后进行磁场计算，得到的磁场也仅仅是相对磁场，
// 再利用 cct345_1_absolute_m 把相对磁场转为绝对磁场
// 此函数中带有大量的魔数，如果修改了机架模型的长度 / 位置，必须做出调整
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
// 2020年11月13日 测试通过
__global__ void test_magnet_cct() {
    int ksi_deg0 = 0;
    int wind_num = 128;
    float bendingAngle = 67.5f;
    float bendingRad = bendingAngle / 180.0f * PI;
    float tiltAngles[TILE_ANGLE_LENGTH] = {30.f, 80.f, 90.f, 90.f};
    float bigR = 0.95f;
    float smallR = 83 * MM + 15 * MM * 2;
    float phi0 = bendingRad / (float) wind_num;
    float current = -9664.f;

    float a = sqrtf(bigR * bigR - smallR * smallR);
    float eta0 = 0.5f * logf((bigR + a) / (bigR - a));
    float ch_eta0 = coshf(eta0);
    float sh_eta0 = sinhf(eta0);

    float k[TILE_ANGLE_LENGTH];

    k[0] = (1.0f / tanf(tiltAngles[0] / 180.0f * PI)) / ((float) (0 + 1) * sh_eta0);
    k[1] = (1.0f / tanf(tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
    k[2] = 0.0;
    k[3] = 0.0;

    float p[3] = {0, 0, 0};
    float m[3] = {0, 0, 0};

    magnet_cct(ksi_deg0, wind_num, phi0, k, a, ch_eta0, sh_eta0, p, m);
//    magnet_cct_parallel(ksi_deg0, wind_num, phi0, k, a, ch_eta0, sh_eta0, p, m);

    printf("device -- %f, %f, %f\n", m[X], m[Y], m[Z]);

    vct_dot_a_v(current * 1e-7f, m);

    // Java --  [0.0031436355039083964, -0.00470478301086915, 0.00888627084434009]
    // device -- 0.003144, -0.004705, 0.008886
    printf("device -- %f, %f, %f\n", m[X], m[Y], m[Z]);
}

// 并发计算每匝线圈磁场，测试成功 2020年11月14日
__global__ void test_magnet_cct_parallel() {
    unsigned int tid_block = threadIdx.x;

    int ksi_deg0 = 0;
    int wind_num = 128;
    float bendingAngle = 67.5f;
    float bendingRad = bendingAngle / 180.0f * PI;
    float tiltAngles[TILE_ANGLE_LENGTH] = {30.f, 80.f, 90.f, 90.f};
    float bigR = 0.95f;
    float smallR = 83 * MM + 15 * MM * 2;
    float phi0 = bendingRad / (float) wind_num;
    float current = -9664.f;

    float a = sqrtf(bigR * bigR - smallR * smallR);
    float eta0 = 0.5f * logf((bigR + a) / (bigR - a));
    float ch_eta0 = coshf(eta0);
    float sh_eta0 = sinhf(eta0);

    float k[TILE_ANGLE_LENGTH];

    k[0] = (1.0f / tanf(tiltAngles[0] / 180.0f * PI)) / ((float) (0 + 1) * sh_eta0);
    k[1] = (1.0f / tanf(tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
    k[2] = 0.0;
    k[3] = 0.0;

    float p[3] = {0, 0, 0};
    __shared__ float m[3];

    magnet_cct_parallel(ksi_deg0, wind_num, phi0, k, a, ch_eta0, sh_eta0, current, p, m);

    // 少一次同步
//    if (tid_block == 0) {
//        vct_dot_a_v(current * 1e-7f, m);
//    }
//    __syncthreads();

    // Java --  [0.0031436355039083964, -0.00470478301086915, 0.00888627084434009]
    if (tid_block == 0) {
        //device -- 0.003144, -0.004705, 0.008886
        printf("device -- %f, %f, %f\n", m[X], m[Y], m[Z]);
    }

}

// 2020年11月13日 测试成功
__global__ void test_particle_run() {
    int ksi_deg0 = 0;
    int wind_num = 128;
    float bendingAngle = 67.5f;
    float bendingRad = bendingAngle / 180.0f * PI;
    float tiltAngles[TILE_ANGLE_LENGTH] = {30.f, 80.f, 90.f, 90.f};
    float bigR = 0.95f;
    float smallR = 83 * MM + 15 * MM * 2;
    float phi0 = bendingRad / (float) wind_num;
    float current = -9664.f;

    float a = sqrtf(bigR * bigR - smallR * smallR);
    float eta0 = 0.5f * logf((bigR + a) / (bigR - a));
    float ch_eta0 = coshf(eta0);
    float sh_eta0 = sinhf(eta0);

    float k[TILE_ANGLE_LENGTH];

    k[0] = (1.0f / tanf(tiltAngles[0] / 180.0f * PI)) / ((float) (0 + 1) * sh_eta0);
    k[1] = (1.0f / tanf(tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
    k[2] = 0.0;
    k[3] = 0.0;

    float p[3] = {0, 0, 0};
    float v[3] = {0.0, 0.0, 1.839551780274753E8};
    float rm = 2.1182873748205775E-27;
    float speed = 1.839551780274753E8;
    float m[3];

    float distance = 0.0f;
    float LENGTH = 1.0f;

    while (distance < LENGTH) {
        // 求磁场
        magnet_cct(ksi_deg0, wind_num, phi0, k, a, ch_eta0, sh_eta0, p, m);
//        magnet_cct_parallel(ksi_deg0, wind_num, phi0, k, a, ch_eta0, sh_eta0, p, m);


        vct_dot_a_v(current * 1e-7f, m);

        // run
        particle_run_step(m, p, v, rm, speed);

        distance += STEP_RUN;
    }

    vct_print(p);
    vct_print(v);
    // last print
    //0.000170, 0.001429, 1.000991
    //-1064.316528, 519703.968750, 183955184.000000
    // Java :
    // position=[1.696032050934739E-4, 0.0014294050422786041, 1.0009986107410314]
    // velocity=[-1064.0616149611867, 519703.2016243721, 1.8395444471527618E8]
}

// 2020年11月14日 测试通过
__global__ void test_particle_run_parallel() {
    unsigned int tid_block = threadIdx.x;

    int ksi_deg0 = 0;
    int wind_num = 128;
    float bendingAngle = 67.5f;
    float bendingRad = bendingAngle / 180.0f * PI;
    float tiltAngles[TILE_ANGLE_LENGTH] = {30.f, 80.f, 90.f, 90.f};
    float bigR = 0.95f;
    float smallR = 83 * MM + 15 * MM * 2;
    float phi0 = bendingRad / (float) wind_num;
    float current = -9664.f;

    float a = sqrtf(bigR * bigR - smallR * smallR);
    float eta0 = 0.5f * logf((bigR + a) / (bigR - a));
    float ch_eta0 = coshf(eta0);
    float sh_eta0 = sinhf(eta0);

    float k[TILE_ANGLE_LENGTH];

    k[0] = (1.0f / tanf(tiltAngles[0] / 180.0f * PI)) / ((float) (0 + 1) * sh_eta0);
    k[1] = (1.0f / tanf(tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
    k[2] = 0.0;
    k[3] = 0.0;

    float p[3] = {0, 0, 0};
    float v[3] = {0.0, 0.0, 1.839551780274753E8};
    float rm = 2.1182873748205775E-27;
    float speed = 1.839551780274753E8;

    // 必须是共享变量
    __shared__ float m[3];

    float distance = 0.0f;
    float LENGTH = 100.0f;

    while (distance < LENGTH) {
        // 求磁场。无需同步
        magnet_cct_parallel(ksi_deg0, wind_num, phi0, k, a, ch_eta0, sh_eta0, current, p, m);
        // run
        particle_run_step(m, p, v, rm, speed);

        distance += STEP_RUN;
    }

    if (tid_block == 0) {
        vct_print(p);
        vct_print(v);
    }
    // 100 m
    // Java position=[-0.002617523327502486, 0.3855863500772864, 100.04325856386859]
    // Java velocity=[-3304.0625271094736, 713319.4063558772, 1.8395379588762134E8]
    // -0.002617, 0.385373, 100.000671
    // -3303.234375, 713905.625000, 183955184.000000


    // 1m
    // last print
    //0.000170, 0.001429, 1.000991
    //-1064.316528, 519703.968750, 183955184.000000
    // Java :
    // position=[1.696032050934739E-4, 0.0014294050422786041, 1.0009986107410314]
    // velocity=[-1064.0616149611867, 519703.2016243721, 1.8395444471527618E8]
}

// 测试相对位置 2020年11月14日 测试通过
__global__ void test_parallel_relative_position() {
    unsigned int tid_block = threadIdx.x;

    int ksi_deg0 = 0;
    int wind_num = 128;
    float bendingAngle = 67.5f;
    float bendingRad = bendingAngle / 180.0f * PI;
    float tiltAngles[TILE_ANGLE_LENGTH] = {30.f, 80.f, 90.f, 90.f};
    float bigR = 0.95f;
    float smallR = 83.f * MM + 15.f * MM * 2.f;
    float phi0 = bendingRad / (float) wind_num;
    float current = -9664.f;

    float a = sqrtf(bigR * bigR - smallR * smallR);
    float eta0 = 0.5f * logf((bigR + a) / (bigR - a));
    float ch_eta0 = coshf(eta0);
    float sh_eta0 = sinhf(eta0);

    float k[TILE_ANGLE_LENGTH];

    k[0] = (1.0f / tanf(tiltAngles[0] / 180.0f * PI)) / ((float) (0 + 1) * sh_eta0);
    k[1] = (1.0f / tanf(tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
    k[2] = 0.0f;
    k[3] = 0.0f;

    float p[3] = {5.0085219608773155, 2.951165121396268, 0.0};
    float v[3] = {0.0f, 0.0f, 1.839551780274753E8f};
    float rm = 2.1182873748205775E-27f;
    float speed = 1.839551780274753E8f;

    float pr[3]; // 相对点
    float m[3]; // 绝对磁场

    __shared__ float mr[3];
    cct345_1_relative_point(p, pr);
    magnet_cct_parallel(ksi_deg0, wind_num, phi0, k, a, ch_eta0, sh_eta0, current, pr, mr);
    cct345_1_absolute_m(mr, m);

    if (tid_block == 0) {
        vct_print(p);
        vct_print(v);
        vct_print(m);
    }

}

// 测试相对位置下粒子运动 2020年11月14日 测试通过
__global__ void test_parallel_relative_position_particle_run() {
    unsigned int tid_block = threadIdx.x;

    int ksi_deg0 = 0;
    int wind_num = 128;
    float bendingAngle = 67.5f;
    float bendingRad = bendingAngle / 180.0f * PI;
    float tiltAngles[TILE_ANGLE_LENGTH] = {30.f, 80.f, 90.f, 90.f};
    float bigR = 0.95f;
    float smallR = 83.f * MM + 15.f * MM * 2.f;
    float phi0 = bendingRad / (float) wind_num;
    float current = -9664.f;

    float a = sqrtf(bigR * bigR - smallR * smallR);
    float eta0 = 0.5f * logf((bigR + a) / (bigR - a));
    float ch_eta0 = coshf(eta0);
    float sh_eta0 = sinhf(eta0);

    float k[TILE_ANGLE_LENGTH];

    k[0] = (1.0f / tanf(tiltAngles[0] / 180.0f * PI)) / ((float) (0 + 1) * sh_eta0);
    k[1] = (1.0f / tanf(tiltAngles[1] / 180.0f * PI)) / ((float) (1 + 1) * sh_eta0);
    k[2] = 0.0f;
    k[3] = 0.0f;

    float p[3] = {3.5121278119986163, 1.45477097251757, 0.0};
    float v[3] = {1.300759538176064E8, 1.3007595381760634E8, 0.0};
    float rm = 2.1182873748205775E-27f;
    float speed = 1.839551780274753E8f;

    float pr[3]; // 相对点
    float m[3]; // 绝对磁场

    __shared__ float mr[3];

    float distance = 0.0f;
    float LENGTH = 2.0f;

    while (distance < LENGTH) {
        // 相对点
        cct345_1_relative_point(p, pr);
        // 相对磁场
        magnet_cct_parallel(ksi_deg0, wind_num, phi0, k, a, ch_eta0, sh_eta0, current, pr, mr);
        // 绝对磁场
        cct345_1_absolute_m(mr, m);

        // run
        particle_run_step(m, p, v, rm, speed);

        distance += STEP_RUN;
    }


    if (tid_block == 0) {
        vct_print(p);
        vct_print(v);
    }

}

int main() {
    test_parallel_relative_position_particle_run<<<128 >>>();
    printf("hello -- host\n");
    return 0;
}

