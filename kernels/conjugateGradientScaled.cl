__kernel void init(int num_vals, __global int *start_index, __global int *end_index,
                __global double *x, __global double *r, __global double *p,
                __constant int *rows, __constant double *b)
{
    int id = get_global_id(0);

    start_index[id] = -1;
    end_index[id] = -1;

    for (int i = id; i < num_vals; i++)
    {
        if ((rows[i] == id) && (start_index[id] == -1))
        {
            start_index[id] = i;
        }
        else if ((rows[i] == id + 1) && (end_index[id] == -1))
        {
            end_index[id] = i - 1;
            break;
        }
        else if ((i == num_vals - 1) && (end_index[id] == -1))
        {
            end_index[id] = i;
        }
    }

    x[id] = 0.0;
    r[id] = b[id];
    p[id] = b[id];
}

__kernel void update_r_length(int dim, __constant double *r, __global double *r_dot_r, __global double *r_length)
{
    r_dot_r[0] = 0.0;
    for (int i = 0; i < dim; i++)
    {
        r_dot_r[0] += r[i] * r[i];
    }
    r_length[0] = sqrt(r_dot_r[0]);
}

__kernel void update_A_times_p(__global double *A_times_p, __constant int *start_index, __constant int *end_index,
                               __constant double *A, __constant double *p, __constant int *cols)
{
    int id = get_global_id(0);

    A_times_p[id] = 0.0;
    for (int i = start_index[id]; i <= end_index[id]; i++)
    {
        A_times_p[id] += A[i] * p[cols[i]];
    }
}

__kernel void calculate_alpha(int dim, __constant double *old_r_dot_r, __constant double *A_times_p, __constant double *p, __global double *alpha)
{
    double Ap_dot_p = 0.0;
    for (int i = 0; i < dim; i++)
    {
        Ap_dot_p += A_times_p[i] * p[i];
    }
    alpha[0] = old_r_dot_r[0]/Ap_dot_p;
}

__kernel void update_guess(__global double *x, __global double *r, __constant double *alpha,
                            __constant double *p, __constant double *A_times_p)
{
    int id = get_global_id(0);

    x[id] += alpha[0] * p[id];
    r[id] -= alpha[0] * A_times_p[id];
}

__kernel void update_direction(__constant double *old_r_dot_r, __constant double *new_r_dot_r, __constant double *r, __global double *p)
{
    int id = get_global_id(0);

    p[id] = r[id] + (new_r_dot_r[0]/old_r_dot_r[0]) * p[id];
}

__kernel void sync_r_dot_r(__global double *old_r_dot_r, __constant double *new_r_dot_r)
{
    old_r_dot_r[0] = new_r_dot_r[0];
}
