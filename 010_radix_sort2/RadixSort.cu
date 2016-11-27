#include "RadixSort.cuh"

__device__ void radixSort(u32* const sort_tmp, u32* const sort_tmp_1, const u32 num_list, const u32 num_elements, const u32 tid)
{
    //Sort into num_list, lists
    //Apply RadixSort on 32 bits of data
    for(u32 bit=0;bit<32;bit++)
    {
        const u32 bit_mask = (1 << bit);
        u32 base_cnt_0 = 0;
        u32 base_cnt_1 = 0;

        for(u32 i=0;i<num_elements;i+=num_list)
        {
            const u32 elem = sort_tmp[i+tid];

            if((elem & bit_mask) > 0)
            {
                sort_tmp_1[base_cnt_1+tid] = elem;
                base_cnt_1+=num_list;
            }
            else
            {
                sort_tmp[base_cnt_0+tid] = elem;
                base_cnt_0+=num_list;
            }
        }

        // Copy data to source from one list
        for(u32 i=0;i<base_cnt_1;i+=num_list)
        {
            sort_tmp[base_cnt_0+i+tid] = sort_tmp_1[i+tid];
        }
    }
    __syncthreads();
}

__device__ void copyDataToShared(u32* const data, u32* const sort_tmp, const u32 num_list, const u32 num_elements, const u32 tid)
{
    // Copy data into temp store (shared memory)
    for(u32 i=0;i<num_elements;i+=num_list)
    {
        sort_tmp[i+tid] = data[i+tid];
    }
    __syncthreads();
}

__device__ void mergeArrays1(const u32 * const src_array, u32* const dest_array, const u32 num_list, const u32 num_elements, const u32 tid)
{
    __shared__ u32 list_indexes[LISTS];

    //Multiple threads
    list_indexes[tid] = 0;
    __syncthreads();

    //Single thread
    if(tid==0)
    {
        const u32 num_elements_per_list = num_elements / num_list;

        for(u32 i=0;i<num_elements;i++)
        {
            u32 min_val = 0xFFFFFFFF;
            u32 min_idx = 0;

            //Iterate over each of the lists
            for(u32 list=0;list<num_list;list++){

                //If current list have already been emptied, then ignore it
                if(list_indexes[list] < num_elements_per_list){
                    const u32 src_idx = list + (list_indexes[list] * num_list);
                    const u32 data = src_array[src_idx];

                    if(data <= min_val){
                        min_val = data;
                        min_idx = list;
                    }
                }
            }
            list_indexes[min_idx]++;
            dest_array[i] = min_val;
        }
    }

}

__global__ void __radixSort__(u32* const data, const u32 num_list, const u32 num_elements)
{
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ u32 sort_tmp[N];
    __shared__ u32 sort_tmp_1[N];

    copyDataToShared(data, sort_tmp, num_list, num_elements, tid);

    radixSort(sort_tmp, sort_tmp_1, num_list, num_elements, tid);

    mergeArrays1(sort_tmp, data, num_list, num_elements, tid);
}

void RadixSort::RunKernelGPU(){
    __radixSort__ <<<BLOCKS, THREADS>>> (d_array, (int)LISTS, (int)N);
}

RadixSort::RadixSort(){
    h_array = (u32*)malloc(N*sizeof(u32));

    srand(time(NULL));

    for(u32 i=0;i<N;i++){
        h_array[i] = rand()%1024;
    }

    cudaMalloc((void**)&d_array, N*sizeof(u32));
    cudaMemcpy(d_array, h_array, N*sizeof(u32), cudaMemcpyHostToDevice);

}

RadixSort::~RadixSort(){
    cudaFree(d_array);
    free(h_array);
}

void RadixSort::CopyResults(){
    cudaMemcpy(h_array, d_array, N*sizeof(u32), cudaMemcpyDeviceToHost);
}
