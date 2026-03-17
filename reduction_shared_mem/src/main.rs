use std::marker::PhantomData;
use std::time::Instant;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use cubecl::frontend::synchronization::sync_cube;

#[derive(Clone)]
pub struct GpuTensor<R: Runtime> {
    pub handle: cubecl::server::Handle,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub _marker: PhantomData<R>,
}

impl<R: Runtime> GpuTensor<R> {
    pub fn arange(shape: Vec<usize>, client: &ComputeClient<R>) -> Self {
        let num_elements: usize = shape.iter().product();
        let data: Vec<f32> = (0..num_elements).map(|i| i as f32).collect();
        let bytes = cubecl::bytes::Bytes::from_elems(data);
        let handle = client.create(bytes);
        
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }

        Self { handle, shape, strides, _marker: PhantomData }
    }

    pub fn empty(shape: Vec<usize>, client: &ComputeClient<R>) -> Self {
        let num_elements: usize = shape.iter().product();
        let data = vec![0.0f32; num_elements];
        let bytes = cubecl::bytes::Bytes::from_elems(data);
        let handle = client.create(bytes);
        
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }

        Self { handle, shape, strides, _marker: PhantomData }
    }

    pub fn as_arg(&self) -> TensorArg<'_, R> {
        unsafe {
            TensorArg::from_raw_parts::<f32>(
                &self.handle,
                &self.strides,
                &self.shape,
                1, 
            ) 
        }
    }
}

// --- Part 2: The Shared Memory Kernel ---

const BLOCK_SIZE: u32 = 256;

#[cube(launch)]
fn reduce_shared_mem(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    let mut shared = SharedMemory::<f32>::new(BLOCK_SIZE as usize);

    let row_idx = CUBE_POS_X as usize;
    let tid = UNIT_POS_X; 

    let mut thread_sum = 0.0;
    
    if row_idx < input.shape(0) {
        let row_offset = row_idx * input.stride(0);
        let num_cols = input.shape(1);
        
        for i in range_stepped(tid, num_cols as u32, BLOCK_SIZE) {
             let idx = row_offset + i as usize;
             thread_sum += input[idx];
        }
    }

    shared[tid as usize] = thread_sum;
    sync_cube();

    // FIX: Replace 'while' with an unrolled 'for' loop.
    // BLOCK_SIZE is 256, so we reduce 8 times (128, 64, 32, 16, 8, 4, 2, 1)
    #[unroll]
    for i in 0..8 {
        // Calculate stride mathematically based on the step
        // i=0 -> shift=7 -> stride=128
        // i=7 -> shift=0 -> stride=1
        let stride = 1 << (7 - i);

        if tid < stride {
            let idx_a = tid as usize;
            let idx_b = (tid + stride) as usize;
            shared[idx_a] += shared[idx_b];
        }
        sync_cube();
    }

    if tid == 0 {
        output[row_idx] = shared[0];
    }
}

// --- Part 3: Benchmark Runner ---

struct SharedBench<R: Runtime> {
    name: String,
    input_shape: Vec<usize>,
    client: ComputeClient<R>,
}

impl<R: Runtime> SharedBench<R> {
    fn new(input_shape: Vec<usize>, client: ComputeClient<R>) -> Self {
        Self {
            name: format!("shared-mem-reduction-{:?}", input_shape),
            input_shape,
            client,
        }
    }

    fn run(&self) {
        println!("Running: {}", self.name);

        let input = GpuTensor::<R>::arange(self.input_shape.clone(), &self.client);
        let output_shape = vec![self.input_shape[0]];
        let output = GpuTensor::<R>::empty(output_shape, &self.client);

        let rows = self.input_shape[0] as u32;

        let cube_count = CubeCount::Static(rows, 1, 1);
        // FIX 4: Use Struct syntax for Dim
        let cube_dim = CubeDim { x: BLOCK_SIZE, y: 1, z: 1 };

        // Warmup
        let _ = reduce_shared_mem::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim.clone(),
            input.as_arg(),
            output.as_arg()
        );
        let _ = pollster::block_on(self.client.sync());

        // Measure
        let start = Instant::now();
        let samples = 100;
        
        for _ in 0..samples {
            let _ = reduce_shared_mem::launch::<R>(
                &self.client,
                cube_count.clone(),
                cube_dim.clone(),
                input.as_arg(),
                output.as_arg()
            );
        }
        
        let _ = pollster::block_on(self.client.sync());
        let duration = start.elapsed();

        println!("  Mean Time: {:?} per run", duration / samples);
    }
}

fn main() {
    let device = Default::default();
    let client = WgpuRuntime::client(&device);

    let bench1 = SharedBench::<WgpuRuntime>::new(vec![512, 8 * 1024], client.clone());
    let bench2 = SharedBench::<WgpuRuntime>::new(vec![128, 32 * 1024], client.clone());

    bench1.run();
    bench2.run();
}