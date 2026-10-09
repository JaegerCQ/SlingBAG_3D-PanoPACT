#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

// One warp per (source, sensor). Only the Gaussian's active time window is visited.
// Forward shards reduce collisions without rounding contributions to fixed point.
__device__ __forceinline__ void time_bounds(float r, float sigma, float h,
    float cutoff, int nt, int &first, int &last) {
    first = 0; last = nt - 1;
    if (cutoff > 0.f) {
        // Conservative extra samples; the exact cutoff mask is applied below.
        float lo = floorf((r - cutoff * sigma) / h) - 1.f;
        float hi = ceilf((r + cutoff * sigma) / h) + 1.f;
        first = static_cast<int>(fminf(float(nt), fmaxf(0.f, lo)));
        last = static_cast<int>(fmaxf(-1.f, fminf(float(nt - 1), hi)));
    }
}

__global__ void forward_kernel(const float *__restrict__ sensors,
    const float *__restrict__ xyz, const float *__restrict__ pressure,
    const float *__restrict__ sigma, float *__restrict__ partial,
    int64_t ns, int64_t n, int nt, float h, float cutoff, int shards) {
    const int lane = threadIdx.x & 31;
    const int64_t pair = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    if (pair >= ns * n) return;
    const int64_t source = pair / ns;
    const int64_t sensor = pair - source * ns;
    const float dx = xyz[3*source] - sensors[3*sensor];
    const float dy = xyz[3*source+1] - sensors[3*sensor+1];
    const float dz = xyz[3*source+2] - sensors[3*sensor+2];
    const float r = sqrtf(dx*dx + dy*dy + dz*dz + 1.e-12f);
    const float width = sigma[source];
    const float inv_sigma = 1.f / width;
    const float weight = pressure[source] / (2.f*r);
    int first, last;
    time_bounds(r, width, h, cutoff, nt, first, last);
    const int shard = (source / 128) % shards;
    float *out = partial + (int64_t(shard)*ns + sensor)*nt;
    for (int t = first + lane; t <= last; t += 32) {
        const float d = r - float(t)*h;
        const float z = d*inv_sigma;
        if (cutoff <= 0.f || fabsf(z) <= cutoff)
            atomicAdd(out+t, weight*d*expf(-.5f*z*z));
    }
}

// One block owns a source. Warps visit different sensors, reduce locally, then
// write each source gradient once. No backward atomics or Ns*N*T tensor.
__global__ void backward_kernel(const float *__restrict__ sensors,
    const float *__restrict__ xyz, const float *__restrict__ pressure,
    const float *__restrict__ sigma, const float *__restrict__ upstream,
    float *__restrict__ grad_xyz, float *__restrict__ grad_p,
    float *__restrict__ grad_sigma, int64_t ns, int64_t n,
    int nt, float h, float cutoff, bool position_grad) {
    const int64_t source = blockIdx.x;
    if (source >= n) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int WARPS = 4;
    const float x = xyz[3*source], y = xyz[3*source+1], zc = xyz[3*source+2];
    const float width = sigma[source];
    const float inv_sigma = 1.f / width;
    const float p = pressure[source];
    float gx=0.f, gy=0.f, gz=0.f, gp=0.f, gs=0.f;
    for (int64_t sensor=warp; sensor<ns; sensor+=WARPS) {
        const float dx=x-sensors[3*sensor], dy=y-sensors[3*sensor+1], dz=zc-sensors[3*sensor+2];
        const float r=sqrtf(dx*dx+dy*dy+dz*dz+1.e-12f);
        const float inv_r=1.f/r;
        int first,last;
        time_bounds(r,width,h,cutoff,nt,first,last);
        float dr=0.f;
        for (int t=first+lane; t<=last; t+=32) {
            const float d=r-float(t)*h;
            const float z=d*inv_sigma;
            if (cutoff <= 0.f || fabsf(z) <= cutoff) {
                const float e=expf(-.5f*z*z);
                const float g=upstream[sensor*nt+t];
                const float common=g*.5f*inv_r*e;
                gp += common*d;
                gs += common*p*d*z*z*inv_sigma;
                if (position_grad) dr += common*p*(1.f-z*z-d*inv_r);
            }
        }
        if (position_grad) {
            gx += dr*dx*inv_r; gy += dr*dy*inv_r; gz += dr*dz*inv_r;
        }
    }
    for (int offset=16; offset>0; offset>>=1) {
        gp += __shfl_down_sync(0xffffffff,gp,offset);
        gs += __shfl_down_sync(0xffffffff,gs,offset);
        gx += __shfl_down_sync(0xffffffff,gx,offset);
        gy += __shfl_down_sync(0xffffffff,gy,offset);
        gz += __shfl_down_sync(0xffffffff,gz,offset);
    }
    __shared__ float values[5][WARPS];
    if (lane==0) {
        values[0][warp]=gx; values[1][warp]=gy; values[2][warp]=gz;
        values[3][warp]=gp; values[4][warp]=gs;
    }
    __syncthreads();
    if (threadIdx.x==0) {
        gx=gy=gz=gp=gs=0.f;
        #pragma unroll
        for (int w=0;w<WARPS;++w) {
            gx+=values[0][w]; gy+=values[1][w]; gz+=values[2][w];
            gp+=values[3][w]; gs+=values[4][w];
        }
        if (position_grad) { grad_xyz[3*source]=gx; grad_xyz[3*source+1]=gy; grad_xyz[3*source+2]=gz; }
        grad_p[source]=gp; grad_sigma[source]=gs;
    }
}

static void check_inputs(const torch::Tensor &s, const torch::Tensor &x,
    const torch::Tensor &p, const torch::Tensor &sigma, double dt, int64_t nt,
    double speed, double cutoff) {
    for (const auto &t: {s,x,p,sigma}) {
        TORCH_CHECK(t.is_cuda() && t.scalar_type()==torch::kFloat32 && t.is_contiguous(),
            "Inputs must be contiguous CUDA float32 tensors");
        TORCH_CHECK(t.device()==s.device(), "Inputs must share a device");
    }
    TORCH_CHECK(s.dim()==2 && s.size(1)==3 && x.dim()==2 && x.size(1)==3,
        "Locations must have shape [N,3]");
    TORCH_CHECK(p.dim()==1 && sigma.dim()==1 && p.numel()==x.size(0) && sigma.numel()==x.size(0),
        "Pressure and sigma must have shape [num_sources]");
    TORCH_CHECK(dt>0 && speed>0 && nt>0 && nt<=INT_MAX && cutoff>=0,
        "Invalid sampling parameters");
}

torch::Tensor gaussian_cuda_forward(torch::Tensor s, torch::Tensor x,
    torch::Tensor p, torch::Tensor sigma, double dt, int64_t nt,
    double speed, double cutoff, int64_t shards) {
    check_inputs(s,x,p,sigma,dt,nt,speed,cutoff);
    c10::cuda::CUDAGuard guard(s.device());
    TORCH_CHECK(shards>=1 && shards<=64,"shards must be in [1,64]");
    auto partial=torch::zeros({shards,s.size(0),nt},s.options());
    const int64_t pairs=s.size(0)*x.size(0);
    if (pairs) {
        constexpr int threads=128;
        const int64_t blocks=(pairs+3)/4;
        TORCH_CHECK(blocks<=INT_MAX,"Too many pairs; split sensors into batches");
        forward_kernel<<<blocks,threads,0,at::cuda::getCurrentCUDAStream()>>>(
            s.data_ptr<float>(),x.data_ptr<float>(),p.data_ptr<float>(),sigma.data_ptr<float>(),
            partial.data_ptr<float>(),s.size(0),x.size(0),nt,float(dt*speed),float(cutoff),shards);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return partial.sum(0);
}

std::vector<torch::Tensor> gaussian_cuda_backward(torch::Tensor s, torch::Tensor x,
    torch::Tensor p, torch::Tensor sigma, torch::Tensor g, double dt, int64_t nt,
    double speed, double cutoff, bool position_grad) {
    check_inputs(s,x,p,sigma,dt,nt,speed,cutoff);
    TORCH_CHECK(g.is_cuda() && g.scalar_type()==torch::kFloat32 && g.is_contiguous() &&
        g.device()==s.device() && g.numel()==s.size(0)*nt,"Invalid upstream gradient");
    c10::cuda::CUDAGuard guard(s.device());
    auto gx=position_grad ? torch::empty_like(x) : torch::zeros_like(x);
    auto gp=torch::empty_like(p), gs=torch::empty_like(sigma);
    if (x.size(0)) {
        backward_kernel<<<x.size(0),128,0,at::cuda::getCurrentCUDAStream()>>>(
            s.data_ptr<float>(),x.data_ptr<float>(),p.data_ptr<float>(),sigma.data_ptr<float>(),
            g.data_ptr<float>(),gx.data_ptr<float>(),gp.data_ptr<float>(),gs.data_ptr<float>(),
            s.size(0),x.size(0),nt,float(dt*speed),float(cutoff),position_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {gx,gp,gs};
}
