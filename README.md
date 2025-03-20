#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WARP_SIZE 32  // CUDA中一个Warp的大小是32个线程
#define MAX_SEQ_LEN 4096  // 支持的最大序列长度

// Warp级归约函数（Butterfly模式）
__device__ float warp_reduce_sum(float val) {
    // 使用Butterfly模式进行Warp级归约
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);  // 使用shfl_down_sync进行线程间数据交换
    return val;  // 返回归约结果
}

// LayerNorm CUDA内核
__global__ void layer_norm_kernel(float4* output, const float4* input, const float* gamma, const float* beta, int seq_len, int hidden_size) {
    // 声明共享内存，用于存储中间结果
    extern __shared__ float shared_mem[];

    // 线程ID和Warp ID
    int tid = threadIdx.x;  // 当前线程的ID
    int warp_id = tid / WARP_SIZE;  // 当前线程所属的Warp ID
    int lane_id = tid % WARP_SIZE;  // 当前线程在Warp中的Lane ID

    // 共享内存布局：
    // shared_data: 存储输入数据的向量化形式（float4）
    // shared_mean: 存储每个Warp的局部均值
    // shared_var: 存储每个Warp的局部方差
    float4* shared_data = (float4*)shared_mem;
    float* shared_mean = (float*)&shared_data[hidden_size / 4];
    float* shared_var = (float*)&shared_mean[blockDim.x / WARP_SIZE];

    // 局部变量：均值和方差
    float mean = 0.0f;
    float var = 0.0f;

    // 计算均值和方差
    for (int i = warp_id; i < seq_len; i += blockDim.x / WARP_SIZE) {
        // 加载输入数据（float4向量化加载）
        float4 data = input[i * (hidden_size / 4) + lane_id];

        // 计算当前Warp的局部和
        float sum = data.x + data.y + data.z + data.w;  // 对float4的4个元素求和
        sum = warp_reduce_sum(sum);  // Warp级归约

        // 将局部和写入共享内存
        if (lane_id == 0) shared_mean[warp_id] = sum;
        __syncthreads();  // 同步所有线程

        // 计算全局均值
        if (tid < WARP_SIZE) {
            float warp_sum = warp_reduce_sum(shared_mean[tid]);  // 对所有Warp的局部和进行归约
            if (tid == 0) shared_mean[0] = warp_sum / (seq_len * hidden_size);  // 计算全局均值
        }
        __syncthreads();  // 同步所有线程

        // 读取全局均值
        mean = shared_mean[0];

        // 计算当前Warp的局部方差
        float diff = (data.x - mean) * (data.x - mean) +
                     (data.y - mean) * (data.y - mean) +
                     (data.z - mean) * (data.z - mean) +
                     (data.w - mean) * (data.w - mean);
        diff = warp_reduce_sum(diff);  // Warp级归约

        // 将局部方差写入共享内存
        if (lane_id == 0) shared_var[warp_id] = diff;
        __syncthreads();  // 同步所有线程

        // 计算全局方差
        if (tid < WARP_SIZE) {
            float warp_diff = warp_reduce_sum(shared_var[tid]);  // 对所有Warp的局部方差进行归约
            if (tid == 0) shared_var[0] = warp_diff / (seq_len * hidden_size);  // 计算全局方差
        }
        __syncthreads();  // 同步所有线程

        // 读取全局方差
        var = shared_var[0];
    }

    // 计算LayerNorm
    for (int i = warp_id; i < seq_len; i += blockDim.x / WARP_SIZE) {
        // 加载输入数据（float4向量化加载）
        float4 data = input[i * (hidden_size / 4) + lane_id];

        // 计算归一化结果
        float4 norm_data;
        norm_data.x = (data.x - mean) / sqrtf(var + 1e-5) * gamma[lane_id * 4] + beta[lane_id * 4];
        norm_data.y = (data.y - mean) / sqrtf(var + 1e-5) * gamma[lane_id * 4 + 1] + beta[lane_id * 4 + 1];
        norm_data.z = (data.z - mean) / sqrtf(var + 1e-5) * gamma[lane_id * 4 + 2] + beta[lane_id * 4 + 2];
        norm_data.w = (data.w - mean) / sqrtf(var + 1e-5) * gamma[lane_id * 4 + 3] + beta[lane_id * 4 + 3];

        // 将结果写回全局内存
        output[i * (hidden_size / 4) + lane_id] = norm_data;
    }
}

// 调用LayerNorm内核
void layer_norm(float4* output, const float4* input, const float* gamma, const float* beta, int seq_len, int hidden_size) {
    // 计算线程块大小和共享内存大小
    int num_warps = (hidden_size + WARP_SIZE - 1) / WARP_SIZE;  // 计算需要的Warp数量
    int block_size = num_warps * WARP_SIZE;  // 线程块大小
    int shared_mem_size = (hidden_size / 4 + 2 * num_warps) * sizeof(float);  // 共享内存大小

    // 启动CUDA内核
    layer_norm_kernel<<<1, block_size, shared_mem_size>>>(output, input, gamma, beta, seq_len, hidden_size);
}
