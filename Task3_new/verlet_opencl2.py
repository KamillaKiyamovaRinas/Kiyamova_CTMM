import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# Гравитационная постоянная
G_SI = 6.67430e-11
SECONDS_PER_MONTH = 2.592e6
G = G_SI * (1e-9) * (SECONDS_PER_MONTH ** 2)

KERNEL_CODE = """
__kernel void compute_accelerations(
    __global const float2* positions,
    __global const float* masses,
    __global float2* accelerations,
    const int N,
    const float G
) {
    int i = get_global_id(0);
    if (i >= N) return;
    
    float2 acc = (float2)(0.0f, 0.0f);
    float2 pos_i = positions[i];
    
    for (int j = 0; j < N; j++) {
        if (i == j) continue;
        
        float2 pos_j = positions[j];
        float2 r_vec = (float2)(pos_j.x - pos_i.x, pos_j.y - pos_i.y);
        float dist_sq = r_vec.x * r_vec.x + r_vec.y * r_vec.y;
        float dist = sqrt(dist_sq);
        
        if (dist < 1e-10f) continue;
        
        float factor = G * masses[j] / (dist * dist * dist);
        acc.x += factor * r_vec.x;
        acc.y += factor * r_vec.y;
    }
    
    accelerations[i] = acc;
}
"""

def init_opencl():
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    program = cl.Program(context, KERNEL_CODE).build()
    return context, queue, program

def compute_accelerations_gpu(positions, masses, queue, program, positions_gpu, masses_gpu, accelerations_gpu):
    N = len(masses)
    
    positions_gpu.set(positions.astype(np.float32))
    kernel = program.compute_accelerations

    kernel.set_arg(0, positions_gpu.data)
    kernel.set_arg(1, masses_gpu.data)
    kernel.set_arg(2, accelerations_gpu.data)
    kernel.set_arg(3, np.int32(N))
    kernel.set_arg(4, np.float32(G))
    
    cl.enqueue_nd_range_kernel(queue, kernel, (N,), None)
    queue.finish()

    return accelerations_gpu.get()

def velocity_verlet_gpu(positions, velocities, masses, t_span, dt):
    t0, t_end = t_span
    steps = int((t_end - t0) / dt)
    N = len(masses)

    context, queue, program = init_opencl()

    positions_gpu = cl_array.to_device(queue, positions.astype(np.float32))
    masses_gpu = cl_array.to_device(queue, masses.astype(np.float32))
    accelerations_gpu = cl_array.to_device(queue, np.zeros_like(positions, dtype=np.float32))

    accelerations = compute_accelerations_gpu(positions, masses, queue, program, 
                                              positions_gpu, masses_gpu, accelerations_gpu)

    trajectory = np.zeros((steps, N, 2), dtype=np.float32)
    
    for step in range(steps):
        trajectory[step] = positions

        positions_new = positions + velocities * dt + 0.5 * accelerations * dt**2
        accelerations_new = compute_accelerations_gpu(positions_new, masses, queue, program,
                                                      positions_gpu, masses_gpu, accelerations_gpu)
        
        velocities_new = velocities + 0.5 * (accelerations + accelerations_new) * dt
        
        positions = positions_new
        velocities = velocities_new
        accelerations = accelerations_new
    
    return trajectory