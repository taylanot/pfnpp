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
  using Tensor = torch::Tensor;

  class Tasks
  {
  public:
    virtual ~Tasks() = default;

    // Pure virtual method to sample tensors
    virtual std::tuple<Tensor, Tensor>
    Sample(int nset, int nsamp, int nfeat) const = 0;
  };

  // Sample datasets from 
  // y = x^t @ w + e
  // w -> 
  // e ~ N(0,a) -> noise
  // x ~ N(0,b) -> input (we must augment this with ones to match the bias)
  // w ~ N(0,c) -> task (includes the offset (bias) as well)
  template<class O = double>
  class LinearTasks final : public Tasks
  {
  public:
    explicit LinearTasks(O a=0, O b=1, O c=1) : a_(a) , b_(b) , c_(c) { }

    std::tuple<Tensor, Tensor>
    Sample(int nset, int nsamp, int nfeat) const override
    {
      Tensor x,w,xs,ys,e;

      w  = torch::normal(0, c_, {nset, nfeat + 1});
      xs = torch::normal(0, b_, {nset, nsamp, nfeat});

      x = torch::cat( { xs, torch::ones({nset, nsamp, 1},
                        xs.options()) }, -1);

      ys = torch::bmm(x, w.unsqueeze(2)).squeeze(-1);
      e  = torch::normal(0, a_, ys.sizes());

      return std::make_tuple( xs.transpose(0, 1),
                              (ys + e).transpose(0, 1).unsqueeze(-1));
    }

  private:
    O a_, b_, c_;
  };
  // Sample datasets from 
  // y = x^t @ w + e
  // w -> 
  // e ~ N(0,a) -> noise
  // x ~ N(0,b) -> input (we must augment this with ones to match the bias)
  // w ~ N(0,c) -> task (includes the offset (bias) as well)
  // libtorch does not have multivariate option so anything you see is isotropic
  /* template<class O=double> */
  /* std::tuple<Tensor, Tensor> linear( int nset, int nsamp, int nfeat, */
  /*                                    O a = 0.0, O b = 1., O c = 1.) */
  /* { */
  /*   Tensor x, w, xs, ys, e; */
  /*   w =  torch::normal(0, c, {nset,nfeat+1}); */
  /*   xs = torch::normal(0, b, {nset,nsamp,nfeat}); */
  /*   x = torch::cat({xs,torch::ones({nset,nsamp,1})}, -1); */ 
  /*   ys = torch::bmm(x,w.unsqueeze(2)).squeeze(-1); */
  /*   e  = torch::normal(0, a, ys.sizes()); */
  /*   return std::make_tuple( xs.transpose(0,1), */
  /*                           (ys+e).transpose(0,1).unsqueeze(-1) ); */
  /* } */

  template<class O = double>
  class LinearSampler
  {
  public:
    explicit LinearSampler(O a_, O b_, O c_)
      : a(a_)
      , b(b_)
      , c(c_)
    {
    }

    std::tuple<Tensor, Tensor> sample(int nset, int nsamp, int nfeat) const
    {
      Tensor x;
      Tensor w;
      Tensor xs;
      Tensor ys;
      Tensor e;

      w  = torch::normal(0, c, {nset, nfeat + 1});
      xs = torch::normal(0, b, {nset, nsamp, nfeat});

      x = torch::cat(
        {
          xs,
          torch::ones({nset, nsamp, 1}, xs.options())
        },
        -1
      );

      ys = torch::bmm(x, w.unsqueeze(2)).squeeze(-1);
      e  = torch::normal(0, a, ys.sizes());

      return std::make_tuple(
        xs.transpose(0, 1),
        (ys + e).transpose(0, 1).unsqueeze(-1)
      );
    }

  private:
    O a;
    O b;
    O c;
  };
}
