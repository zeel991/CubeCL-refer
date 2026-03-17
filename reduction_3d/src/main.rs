use std::marker::PhantomData;
use std::time::Instant;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

// --- Part 1: Helper Struct (Vectorized) ---

#[derive(Clone)]
pub struct GpuTensor<R: Runtime> {
    pub handle: cubecl::server::Handle,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub _marker: PhantomData<R>,
}

impl<R: Runtime> GpuTensor<R> {
    pub fn new(shape: Vec<usize>, client: &ComputeClient<R>, fill_value: Option<f32>) -> Self {
        let num_elements: usize = shape.iter().product();
        let data: Vec<f32> = match fill_value {
            Some(v) => vec![v; num_elements],
            None => (0..num_elements).map(|i| (i % 100) as f32).collect(),
        };
        
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
                vectorization as usize, 
            ) 
        }
    }
}

// --- Part 2: The 3D Reduction Kernel ---

const VECTOR_SIZE: u32 = 4;

#[cube(launch)]
fn reduce_3d(
    input: &Tensor<Line<f32>>, 
    output: &mut Tensor<Line<f32>>
) {
    // Strategy:
    // Dimension 0 (X) -> Managed by Block ID (CUBE_POS_X)
    // Dimension 1 (Y) -> Managed by Thread ID (UNIT_POS_X)
    // Dimension 2 (Z) -> Reduced by Loop
    
    // We accumulate vectors directly
    let mut acc = Line::new(0.0);
    
    let z_dim = input.shape(2);
    let vector_stride = input.stride(0); // This stride is tricky, let's calculate manually below
    
    // Calculate memory offset
    // In a 3D tensor [X, Y, Z], the linear index is:
    // idx = x * stride_x + y * stride_y + z * stride_z
    
    // Note: input.stride(0) usually gives stride for dim 0.
    // Let's rely on standard stride logic provided by the tensor.
    let x_id = CUBE_POS_X as usize;
    let y_id = UNIT_POS_X as usize;

    let x_offset = x_id * input.stride(0);
    let y_offset = y_id * input.stride(1);
    
    // The Z-dimension is the contiguous one (stride(2) is usually 1, or 1/4 for vectors)
    // We loop through the depth (Z)
    let num_vectors_z = z_dim / ( VECTOR_SIZE as usize);

    for z in 0..num_vectors_z {
        // Since we are vectorized, 'z' steps by 1 vector (4 floats)
        // input.stride(1) handles the jump between rows
        // We just need to add 'z' because it's the fastest moving dimension
        let idx = x_offset + y_offset + z;
        acc += input[idx];
    }

    // Write output [X, Y]
    // Output is 2D, so stride(0) corresponds to X
    let out_idx = x_id * output.stride(0) + y_id;
    output[out_idx] = acc;
}

// --- Part 3: Benchmark Runner ---

struct Bench3D<R: Runtime> {
    name: String,
    input_shape: Vec<usize>,
    client: ComputeClient<R>,
}

impl<R: Runtime> Bench3D<R> {
    fn new(input_shape: Vec<usize>, client: ComputeClient<R>) -> Self {
        Self {
            name: format!("3d-reduction-{:?}", input_shape),
            input_shape,
            client,
        }
    }

    fn run(&self) {
        println!("Running: {}", self.name);

        let input = GpuTensor::<R>::new(self.input_shape.clone(), &self.client, None);
        
        // Output reduces the last dimension: [X, Y, Z] -> [X, Y]
        let output_shape = vec![self.input_shape[0], self.input_shape[1]];
        let output = GpuTensor::<R>::new(output_shape, &self.client, Some(0.0));

        // MAPPING:
        // CUBE_COUNT (X) = Input Dim 0 (64)
        // CUBE_DIM   (X) = Input Dim 1 (256)
        let cube_count = CubeCount::Static(self.input_shape[0] as u32, 1, 1);
        let cube_dim = CubeDim { x: self.input_shape[1] as u32, y: 1, z: 1 };

        // Warmup
        let _ = reduce_3d::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim.clone(),
            input.as_arg(VECTOR_SIZE as u8),
            output.as_arg(VECTOR_SIZE as u8)
        );
        let _ = pollster::block_on(self.client.sync());

        // Measure
        let start = Instant::now();
        let samples = 100;
        
        for _ in 0..samples {
            let _ = reduce_3d::launch::<R>(
                &self.client,
                cube_count.clone(),
                cube_dim.clone(),
                input.as_arg(VECTOR_SIZE as u8),
                output.as_arg(VECTOR_SIZE as u8)
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

    // Case 1: [64, 256, 1024] -> Reduces to [64, 256]
    // 64 blocks, 256 threads each. Each thread sums 1024/4 = 256 vectors.
    let bench1 = Bench3D::<WgpuRuntime>::new(vec![64, 256, 1024], client.clone());
    
    // Case 2: [64, 64, 4096] -> Reduces to [64, 64]
    // 64 blocks, 64 threads each. Each thread sums 4096/4 = 1024 vectors.
    let bench2 = Bench3D::<WgpuRuntime>::new(vec![64, 64, 4096], client.clone());

    bench1.run();
    bench2.run();
}