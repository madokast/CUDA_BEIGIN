// 计算一层 CCT 在 p 点产生的磁场。注意磁场还要再乘电流 和 μ0/4π (=1e-7)
// ksi_deg0 是计算的起点。wind_num 是匝数。
// phi0、k_tilt_angles、a、ch_eta0、sh_eta0 见 point_cct 函数，p 点为需要计算磁场的点，m_ret 是返回的磁场。
__device__ void // 不再使用，粒度太大
magnet_cct(int ksi_deg0, int wind_num, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0,
           float *p, float *m_ret);


__device__ void
magnet_cct_parallel(int ksi_deg0, int wind_num, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0,
                    float current, float *p, /*__shared__*/ float *m_ret);


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
__device__  void // 此函数粒度不够，不再使用
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

// 2020年11月13日 测试通过
__global__ void test_magnet_cct()
{
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


    if (tid < CCT_TOTAL_WIND_NUM) {
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

__global__ void test_particle_run_parallel(float *data) {
    unsigned int tid = threadIdx.x;

    float p[3] = {3.5121278119986163, 1.45477097251757, 0.0};
    float v[3] = {1.2326128074269669E8, 1.2326128074269663E8, 0.0};
    float rm = 2.0558942080656965E-27;
    float speed = 1.7431777494179922E8;

    float distance = 0.0f;
    float LENGTH = 3.433224382841364f;

    float pr_cct[2][3]; // 相对点
    float m[2][3]; // 绝对磁场

    __shared__ float mr[2][3]; // 相对磁场

    while (distance < LENGTH) {
        // 相对点
        cct345_1_relative_point(p, pr_cct[0]);
        // 磁场
        magnet_at_cct345_1(data, pr_cct[0], mr[0]);
        // 绝对磁场
        cct345_1_absolute_m(mr[0], m[0]);
        // 粒子运动
        particle_run_step(m[0], p, v, rm, speed);

        distance += STEP_RUN;
    }

    if (tid == 0) {
        // java
        // {position=[3.5121278119986163, 1.45477097251757, 0.0],
        // velocity=[1.2326128074269669E8, 1.2326128074269663E8, 0.0], distance=0.0}
        // {position=[6.225937108498759, 3.0639867524971423, -0.001508755653133722],
        // velocity=[1.595159970266312E8, -7.054897849029969E7, -279805.24817167915], distance=3.433224382841577}

        // cuda
        // 6.226668, 3.063664, -0.001510
        //159516464.000000, -70547552.000000, -279971.218750

        // diff
        // ([-7.30891501e-04,  3.22752497e-04,  1.24434687e-06]) 单位 m
        // ([-0.7308915 ,  0.3227525 ,  0.00124435]) 单位 mm
        vct_print(p);
        vct_print(v);
        printf("%f\n", distance); // 3.433933
    }
}

// 测试 CCT345_2 磁场，成了
__global__ void test_cct345_2_magnet(float *data) {
    unsigned int tid = threadIdx.x;

    float p[3] = {7.157070170396251, 2.366002833170567, 0.0};

    float pr_cct[2][3]; // 相对点
    float m[2][3]; // 绝对磁场

    __shared__ float mr[2][3]; // 相对磁场

    cct345_2_relative_point(p, pr_cct[1]);

    magnet_at_cct345_1(data, pr_cct[1], mr[1]);

    cct345_2_absolute_m(mr[1], m[1]);

    if (tid == 0) {
        // Java
        // -0.00979169237915073, 0.027465786414956783, 2.403755237120381
        // cuda
        // -0.009778, 0.027427, 2.403643
        vct_print(m[1]);
    }
}

__global__ void test_cct345_1_and_2_magnet(float *data) {
    unsigned int tid = threadIdx.x;

    float p[3] = {6.336646882102519, 3.035807472522424, 0.0};

    float pr_cct[2][3]; // 相对点
    float m[2][3]; // 绝对磁场

    __shared__ float mr[2][3]; // 相对磁场

    cct345_1_relative_point(p, pr_cct[0]);
    cct345_2_relative_point(p, pr_cct[1]);

    magnet_at_cct345_1_and_2(data, pr_cct, mr);

    cct345_1_absolute_m(mr[0], m[0]);
    cct345_2_absolute_m(mr[1], m[1]);

    if (tid == 0) {
        vct_print(m[1]);
        vct_print(m[0]);

        vct_add_local(m[0], m[1]);
        // java
        //
        // cuda
        //
        vct_print(m[0]);
    }
}

// 测试 qs3 ，测试通过 2020年11月16日
__global__ void test_magnet_at_qs3(float *h_data) {
    float p0[3] = {6.235815085670027, 3.0775733701232055, 0.0};
    float m0[3];

    magnet_at_qs3(-7.3733f, -45.31f * 2, p0, m0);
    if (threadIdx.x == 0) {
        vct_print(m0);
    }

    //-------------------

    // new point
    float p1[3] = {6.239641919993678, 3.0868121654483183, 0.0};
    float m1[3];

    magnet_at_qs3(200, 0.0 * 2, p1, m1);
    if (threadIdx.x == 0) {
        // -0.000000, -0.000000, 2.000050
        // 0.0, 0.0, 2.000000000000078
        vct_print(m1);
    }

    // new point
    float p2[3] = {6.235815085670027, 3.0775733701232055, 0.01};
    float m2[3];

    magnet_at_qs3(200, 0.0 * 2, p2, m2);
    if (threadIdx.x == 0) {
        // 0.765367, 1.847759, 0.000000
        // 0.7653668647301795, 1.8477590650225735, -0.0
        vct_print(m2);
    }

    // new point
    float p3[3] = {6.235815085670027, 3.0775733701232055, 0.01};
    float m3[3];

    magnet_at_qs3(0.0, 200 * 2, p3, m3);
    if (threadIdx.x == 0) {
        // 0.000000, 0.000000, -0.020000
        // -0.0, -0.0, -0.02
        vct_print(m3);
    }

    // new point
    float p4[3] = {6.235815085670027, 3.0775733701232055, 0.01};
    float m4[3];

    magnet_at_qs3(0.0, 200 * 2, p4, m4);
    if (threadIdx.x == 0) {
        // 0.000000, 0.000000, -0.020000
        // 0.0, 0.0, 0.02000000000000156
        vct_print(m4);
    }

    // new point
    float p5[3] = {6.34985075343268, 3.0565930066029123, 0.07383321438269272};
    float m5[3];

    magnet_at_qs3(10, 10000 * 2, p5, m5);
    if (threadIdx.x == 0) {
        // 13.989565, 33.773796, -48.387268
        // 13.989622058936229, 33.77393530715767, -48.38722507873593
        vct_print(m5);
    }

}