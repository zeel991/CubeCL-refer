use std::marker::PhantomData;
use std::time::Instant;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

// --- Part 1: Helper Struct for GPU Data ---

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
        
        // FIX: Use 'from_elems' to convert Vec<f32> directly to Bytes
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
        
        // FIX: Use 'from_elems' here too
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
// --- Part 2: The Parallel Kernel ---

#[cube(launch)]
fn reduce_matrix_parallel(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    // Cast to usize for array indexing compatibility
    let row_index = ABSOLUTE_POS_X as usize;

    // Safety Check
    if row_index < input.shape(0) {
        let mut acc = 0.0;

        for j in 0..input.shape(1) {
            // Ensure index calculation is all in usize
            let idx = row_index * input.stride(0) + j;
            acc += input[idx];
        }

        output[row_index] = acc;
    }
}

// --- Part 3: Benchmark Runner ---

struct ParallelBench<R: Runtime> {
    name: String,
    input_shape: Vec<usize>,
    client: ComputeClient<R>,
}

impl<R: Runtime> ParallelBench<R> {
    fn new(input_shape: Vec<usize>, client: ComputeClient<R>) -> Self {
        Self {
            name: format!("parallel-reduction-{:?}", input_shape),
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

        // v0.9 Launch Config
        let cube_count = CubeCount::Static(rows, 1, 1);
        let cube_dim = CubeDim { x: 1, y: 1, z: 1 };

        // Warmup
        let _ = reduce_matrix_parallel::launch::<R>(
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
            let _ = reduce_matrix_parallel::launch::<R>(
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

    let bench1 = ParallelBench::<WgpuRuntime>::new(vec![512, 8 * 1024], client.clone());
    let bench2 = ParallelBench::<WgpuRuntime>::new(vec![128, 32 * 1024], client.clone());

    bench1.run();
    bench2.run();
}