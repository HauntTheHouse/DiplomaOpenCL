__kernel void steepestDescent(int dim, int num_vals, __local double *r, __local double* A_times_r,
                              __global int *rows, __global int *cols, __global double *A, __global double *b,
                              __global double *x, __global double *result)
{
    int id = get_local_id(0);

    local double alpha, r_length;
    local int iteration;

    int start_index = -1;
    int end_index = -1;
    double r_dot_r, Ar_dot_r;

    // find matrix indices
    for (int i = id; i < num_vals; i++)
    {
        if ((rows[i] == id) && (start_index == -1))
        {
            start_index = i;
        }
        else if ((rows[i] == id + 1) && (end_index == -1))
        {
            end_index = i - 1;
            break;
        }
        else if ((i == num_vals - 1) && (end_index == -1))
        {
            end_index = i;
        }
    }

    // set initial guess / residual
    r[id] = b[id];
    x[id] = 0.0;

    barrier(CLK_LOCAL_MEM_FENCE);

    iteration = 0;
    r_length = 0.01;
    while (iteration < 50000 && r_length >= 0.01)
    {
        A_times_r[id] = 0.0;
        for (int i = start_index; i <= end_index; i++)
        {
            A_times_r[id] += A[i] * r[cols[i]];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (id == 0)
        {
            // compute alpha
            r_dot_r = 0.0;
            Ar_dot_r = 0.0;
            for (int i = 0; i < dim; i++)
            {
                r_dot_r += r[i] * r[i];
                Ar_dot_r += A_times_r[i] * r[i];
            }
            alpha = r_dot_r/Ar_dot_r;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // update guess/residual
        x[id] += alpha * r[id];
        r[id] -= alpha * A_times_r[id];

        barrier(CLK_LOCAL_MEM_FENCE);

        if (id == 0)
        {
            r_length = sqrt(r_dot_r);
            iteration++;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result[0] = (double)iteration;
    result[1] = r_length;
}