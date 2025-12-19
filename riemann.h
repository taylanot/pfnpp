/*
  * Author: Ozgur Taylan Turan
  * Date: 16 December 2025
  * Description: This is the Riemann Distribution mentioned in [1].
  *
*/

namespace dist 
{
  using namespace torch;

  template<class O=double>
  struct Riemann : nn::Module
  {
    Riemann ( const Tensor& bins, bool ignore = true ) : 
      bins_(bins), ignore_(ignore)
    {
      TORCH_CHECK( bins.dim() == 1, "Expecting a 1D Tensor..." );
      bins_ = register_buffer("bins_", std::get<0>(sort(bins)).contiguous());
      auto widths = _bucket_widths( );
      Tensor width = widths.sum();
      this -> to(DEVICE);
    };

    // Get all bucket widths
    Tensor _bucket_widths( ) const 
    {
      using namespace indexing;
      return bins_.index({Slice(1,None)}) - bins_.index({Slice(None,-1)});
    };

    // Find which bin the outputs fall
    Tensor _map( const Tensor& y ) const
    {
      auto res = searchsorted(bins_,y)-1;
      // these are for the boundery values and the more extremee values observed
      res.index_put_({y == bins_.index({0})}, 0);
      res.index_put_({y == bins_.index({-1})}, _nbins() - 1);
      res = torch::clamp(res, 0, _nbins() - 1);  // clamp to valid bin range
      return res;
    }

    // Where to ignore? Let op; your labels are now adjusted;
    Tensor _ignore( Tensor& y ) const
    {
      auto where = y.isnan();
      TORCH_CHECK ( !(where.any().item<bool>() && !ignore_) ,
          "You have nan's. If you want to ignore do it explicetly!" )
      // Put all the nan's to the borders...
      y.index_put_({where}, bins_[0]);  
      return where;
    }

    // Get the number of bins
    int _nbins( ) const 
    {
      return bins_.numel() - 2;
    };

    Tensor forward(const Tensor& logits, const Tensor& y)
    {
      auto y_ = y.clone().contiguous();

      Tensor ignore_mask = _ignore(y_);

      auto logits_ = logits.view({-1, logits.size(2)});

      Tensor target = _map(y_).view(-1);

      nn::CrossEntropyLoss crs;

      auto res = crs(logits_, target);
      return res;
    }

    torch::Tensor mean(const torch::Tensor& logits)
    {
      auto bucket_means = bins_.slice(0, 0, -1) + _bucket_widths() / 2.0;
      return torch::matmul(torch::softmax(logits, -1), bucket_means);
    }

    bool ignore_;
    Tensor bins_;

  };

template<class O=double>
Tensor bin_borders( int num_outputs,
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


}






