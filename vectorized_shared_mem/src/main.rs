use std::marker::PhantomData;
use std::time::Instant;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use cubecl::frontend::synchronization::sync_cube;

// --- Part 1: Helper Struct (Updated for Vectorization) ---

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

    // FIX: Re-added vectorization factor argument
    pub fn as_arg(&self, vectorization: u8) -> TensorArg<'_, R> {
        unsafe {
            TensorArg::from_raw_parts::<f32>(
                &self.handle,
                &self.strides,
                &self.shape,
                vectorization as usize, 
            ) 
        }
    }
}

// --- Part 2: The Vectorized Shared Memory Kernel ---

const BLOCK_SIZE: u32 = 256;
const VECTOR_SIZE: u32 = 4;

#[cube(launch)]
fn reduce_shared_vectorized(
    input: &Tensor<Line<f32>>, // Vectorized Input
    output: &mut Tensor<f32>   // Scalar Output
) {
    // 1. Setup Shared Memory (Scalar f32 is fine here)
    let mut shared = SharedMemory::<f32>::new(BLOCK_SIZE as usize);

    let row_idx = CUBE_POS_X as usize;
    let tid = UNIT_POS_X; 

    // 2. Vectorized Loading Loop
    // Initialize a VECTOR accumulator [0.0, 0.0, 0.0, 0.0]
    let mut vector_acc = Line::new(0.0);
    
    if row_idx < input.shape(0) {
        let row_offset = row_idx * input.stride(0); // Element offset
        let num_cols = input.shape(1);
        
        // Adjust loop for vector size!
        // We iterate 'num_cols / 4' times
        let num_vectors = num_cols / VECTOR_SIZE as usize;
        let vector_stride = input.stride(0) / VECTOR_SIZE as usize;

        // Note: range_stepped step is BLOCK_SIZE, meaning each thread handles
        // one vector every 256 vectors.
        for i in range_stepped(tid, num_vectors as u32, BLOCK_SIZE) {
             let idx = row_idx * vector_stride + i as usize;
             vector_acc += input[idx]; // Vector Load + Add
        }
    }

    // 3. Horizontal Reduction (Vector -> Scalar)
    // Sum the 4 components of the vector into one scalar
    let mut thread_sum = 0.0;
    #[unroll]
    for i in 0..4 {
        thread_sum += vector_acc[i];
    }

    // 4. Store to Shared Memory & Sync
    shared[tid as usize] = thread_sum;
    sync_cube();

    // 5. Standard Shared Memory Tree Reduction (Unrolled)
    #[unroll]
    for i in 0..8 {
        let stride = 1 << (7 - i);
        if tid < stride {
            let idx_a = tid as usize;
            let idx_b = (tid + stride) as usize;
            shared[idx_a] += shared[idx_b];
        }
        sync_cube();
    }

    // 6. Write Result
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
            name: format!("vectorized-shared-reduction-{:?}", input_shape),
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
        let cube_dim = CubeDim { x: BLOCK_SIZE, y: 1, z: 1 };

        // Warmup
        let _ = reduce_shared_vectorized::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim.clone(),
            // Pass vectorization factor 4 here!
            input.as_arg(VECTOR_SIZE as u8),
            output.as_arg(1)
        );
        let _ = pollster::block_on(self.client.sync());

        // Measure
        let start = Instant::now();
        let samples = 100;
        
        for _ in 0..samples {
            let _ = reduce_shared_vectorized::launch::<R>(
                &self.client,
                cube_count.clone(),
                cube_dim.clone(),
                input.as_arg(VECTOR_SIZE as u8),
                output.as_arg(1)
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