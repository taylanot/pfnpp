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

    Tensor _Bins( int num_outputs,
                  const c10::optional<torch::Tensor>& full_range,
                  const c10::optional<torch::Tensor>& ys )
    {
      TORCH_CHECK( ys.has_value() != full_range.has_value(),
        "Either ys or full_range must be passed, but not both." );

      torch::Tensor borders;

      if (ys.has_value())
      {
        torch::Tensor y = ys.value().flatten();
        y = y.masked_select(~torch::isnan(y));

        TORCH_CHECK( y.numel() > num_outputs,
          "Number of ys must be larger than num_outputs." );

        torch::Tensor quantiles = torch::linspace( 0.0, 1.0, num_outputs + 1);

        borders = torch::quantile(y, quantiles);

        if (full_range.has_value())
        {
          TORCH_CHECK( full_range->numel() == 2,
            "full_range must be a tensor of size 2." );

          borders.index_put_({0}, (*full_range)[0]);
          borders.index_put_({-1}, (*full_range)[1]);
        }
      }
      else
      {
        TORCH_CHECK( full_range->numel() == 2,
          "full_range must be a tensor of size 2.");

        borders =
          torch::linspace( full_range.value()[0], full_range.value()[1], 
                           num_outputs + 1 );
      }

      borders = std::get<0>(torch::unique_consecutive(borders));

      TORCH_CHECK( borders.numel() - 1 == num_outputs,
        "len(borders) - 1 must equal num_outputs." );

      return borders;
    }

    Tensor Border( int nsamp, int nfeat, int nbin )
    {
      auto res = this->Sample(100000, nsamp, nfeat);
      return this-> _Bins(nbin, c10::nullopt, std::get<1>(res));
    }
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
