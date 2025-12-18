/*
  * Author: Ozgur Taylan Turan
  * Date: 17 December 2025
  * Description: This is for all the priors you might want to train on ...
  *
*/
#pragma once
#include <tuple>

namespace prior
{
  using namespace torch;

  // Sample datasets from 
  // y = x^t @ w + e
  // w -> 
  // e ~ N(0,a) -> noise
  // x ~ N(0,b) -> input (we must augment this with ones to match the bias)
  // w ~ N(0,c) -> task (includes the offset (bias) as well)
  // libtorch does not have multivariate option so anything you see is isotropic
  template<class O=double>
  std::tuple<Tensor, Tensor> linear( int nset, int nsamp, int nfeat,
                                     O a = 0.0, O b = 1./12., O c = 1.)
  {
    // samplings are problematic without namespaces
    Tensor x, w, xs, ys, e;
    xs = torch::normal(0, b, {nset,nsamp,nfeat});
    x = torch::cat({xs,torch::ones({nset,nsamp,1})}, -1); 
    w =  torch::normal(0, c, {nset,nfeat+1});
    ys = torch::bmm(x,w.unsqueeze(2)).squeeze(-1);
    e  = torch::normal(0, a, ys.sizes());
    return std::make_tuple(xs, (ys+e).unsqueeze(-1));
  }
}
