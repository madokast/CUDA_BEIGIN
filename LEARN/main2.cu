__global__ void particle_run_parallel(float *data, float *qs_data) {
    // TODO 增加粒子输入、输出、多机架，必成！
    int i;
    unsigned int tid = threadIdx.x;

    float qs_q = qs_data[0];
    float qs_s = qs_data[1];

    float ps[] = {3.5121278119986163, 1.45477097251757, 0.0, 3.5121278119986163, 1.45477097251757, 0.0};
    float vs[] = {1.2326128074269669E8, 1.2326128074269663E8, 0.0, 1.2326128074269669E8, 1.2326128074269663E8, 0.0};

    float p[3];
    float v[3];
    float rm = 2.0558942080656965E-27;
    float speed = 1.7431777494179922E8;

    float distance;
    float LENGTH = 7.104727865682728f;

    float pr_cct[2][3]; // 相对点
    float m[2][3]; // 绝对磁场

    __shared__ float mr[2][3]; // 相对磁场

    for(i=0;i<2;i++){
        distance = 0.0f;
        vct_copy(ps+i*3,p);
        vct_copy(vs+i*3,v);

        while (distance < LENGTH) {
            // 相对点
            cct345_1_relative_point(p, pr_cct[0]);
            cct345_2_relative_point(p, pr_cct[1]);
            // 磁场
            magnet_at_cct345_1_and_2(data, pr_cct, mr);
            // 绝对磁场
            cct345_1_absolute_m(mr[0], m[0]);
            cct345_2_absolute_m(mr[1], m[1]);

            // 总绝对磁场，再加上 qs3 的磁场
            vct_add_local(m[0], m[1]);
            magnet_at_qs3(qs_q, qs_s, p, m[1]);
            vct_add_local(m[0], m[1]);

            // 粒子运动
            particle_run_step(m[0], p, v, rm, speed);

            distance += STEP_RUN;
        }

        if (tid == 0) {
            vct_print(p);
            vct_print(v);
            printf("%f", distance);
        }
    }
}