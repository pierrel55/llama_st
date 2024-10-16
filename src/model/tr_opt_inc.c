// this file must be included in transformer.c if USE_THRD_BATCH defined.
// batch some tread work, allow to gain about 2 to 6 % speed

#ifdef INC_THRD_BATCH

// define qkv for self attention
static _inline void opt_compute_qkv(float *q, float *k, float *v, const float *xb, const struct transformer_weights_t *w, int layer_id, mm_proc_t matmul_lw)
{
  int n_thrd = numa_map.n_threads;
  int i;
  CHECK(n_thrd <= w->wk.wy);

  #pragma omp parallel for
  for (i=0; i<n_thrd; i++)
  {
    const struct w_part_t *lp;
    const char *_p;
    int y = i*w->wk.dy;
    int dy = WD_GET_DY(y, w->wk.dy, w->wk.wy);
    size_t ofs;

    lp = &w->wk.lp[i];
    ofs = (size_t)layer_id * lp->sz_l;   // same for k and v  (same wy)
    _p = (const char *)lp->p + ofs;
    matmul_lw(k + y, xb, _p, w->wk.wx, dy);

    lp = &w->wv.lp[i];
    _p = (const char *)lp->p + ofs;
    matmul_lw(v + y, xb, _p, w->wv.wx, dy);

    if (q)
    {
      y = i*w->wq.dy;
      dy = WD_GET_DY(y, w->wq.dy, w->wq.wy);
      lp = &w->wq.lp[i];
      _p = (const char *)lp->p + (size_t)layer_id * lp->sz_l;
      matmul_lw(q + y, xb, _p, w->wq.wx, dy);
    }
  }
}

// in xb, work hb2, out hb
static _inline void opt_compute_w1_w3_swiglu(float *hb, float *hb2, const float *xb, const struct transformer_weights_t *w, int layer_id, mm_proc_t matmul_lw)
{
  int n_thrd = numa_map.n_threads;
  int i;
  CHECK(n_thrd <= w->w1.wy);

  #pragma omp parallel for
  for (i=0; i<n_thrd; i++)
  {
    const struct w_part_t *lp;
    const char *_p;
    int y = i*w->w1.dy;
    int dy = WD_GET_DY(y, w->w1.dy, w->w1.wy);
    int wx = w->w1.wx;
    int x, x1;
    size_t ofs;

    lp = &w->w1.lp[i];
    ofs = (size_t)layer_id * lp->sz_l;   // same for w1/w3 (same wy)
    _p = (const char *)lp->p + ofs;
    matmul_lw(hb + y, xb, _p, wx, dy);

    lp = &w->w3.lp[i];
    _p = (const char *)lp->p + ofs;
    matmul_lw(hb2 + y, xb, _p, wx, dy);

    // swiglu
    x1 = y+dy;
    for (x=y; x<x1; x++)
      hb[x] = swiglu(hb[x]) * hb2[x];
  }
}

#endif
