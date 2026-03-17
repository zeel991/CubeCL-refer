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

    pub fn as_arg(&self, vectorization: u8) -> TensorArg<'_, R> {
        unsafe {
            TensorArg::from_raw_parts::<f32>(
                &self.handle,
                &self.strides,
                &self.shape,
                // FIX 1: Explicit cast to usize
                vectorization as usize, 
            ) 
        }
    }
}

// --- Part 2: The Vectorized Kernel ---

const VECTOR_SIZE: u32 = 4;

#[cube(launch)]
fn reduce_matrix_vectorized(
    input: &Tensor<Line<f32>>, 
    output: &mut Tensor<f32>
) {
    let row_index = ABSOLUTE_POS_X as usize;

    if row_index < input.shape(0) {
        // Line::new(val) creates a vector [val, val, val, val]
        let mut acc = Line::new(0.0);

        let line_stride = input.stride(0) / VECTOR_SIZE as usize;
        let num_lines = input.shape(1) / VECTOR_SIZE as usize;

        for j in 0..num_lines {
            let idx = row_index * line_stride + j;
            acc += input[idx];
        }

        // FIX 2 & 3: Manual horizontal reduction
        // Instead of .sum(), we manually sum the 4 vector components.
        // This is the most compatible way across all backends.
        let mut sum = 0.0;
        #[unroll] // Hint compiler to unroll this tiny loop
        for i in 0..4 {
            sum += acc[i];
        }

        output[row_index] = sum;
    }
}

// --- Part 3: Benchmark Runner ---

struct VectorizedBench<R: Runtime> {
    name: String,
    input_shape: Vec<usize>,
    client: ComputeClient<R>,
}

impl<R: Runtime> VectorizedBench<R> {
    fn new(input_shape: Vec<usize>, client: ComputeClient<R>) -> Self {
        Self {
            name: format!("vectorized-reduction-{:?}", input_shape),
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
        let cube_dim = CubeDim { x: 1, y: 1, z: 1 };

        let _ = reduce_matrix_vectorized::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim.clone(),
            input.as_arg(VECTOR_SIZE as u8),
            output.as_arg(1)
        );
        let _ = pollster::block_on(self.client.sync());

        let start = Instant::now();
        let samples = 100;
        
        for _ in 0..samples {
            let _ = reduce_matrix_vectorized::launch::<R>(
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

    let bench1 = VectorizedBench::<WgpuRuntime>::new(vec![512, 8 * 1024], client.clone());
    let bench2 = VectorizedBench::<WgpuRuntime>::new(vec![128, 32 * 1024], client.clone());

    bench1.run();
    bench2.run();
}