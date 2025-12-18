/*
  * Author: Ozgur Taylan Turan
  * Date: 16 December 2025
  * Description: Collect the models here.
  *
*/

namespace model
{
  

  struct SimplePFN : torch::nn::TransformerImpl
  {

    torch::nn::Linear embedx{nullptr}, embedy{nullptr};

    int dmodel_, nhead_, nencoder_, nhid_, infeat_, nbin_;

    SimplePFN( int dmodel, int nhead, int nencoder, int nhid, int infeat=1,
               int nbin = 100 )

    : torch::nn::TransformerImpl( torch::nn::TransformerOptions(dmodel, nhead)
                          .num_encoder_layers(nencoder)
                          .dim_feedforward(nhid)
                          .dropout(0.) )
    , dmodel_(dmodel)
    , nhead_(nhead)
    , nencoder_(nencoder)
    , nhid_(nhid)
    , infeat_(infeat)
    , nbin_(nbin)

    {
      embedx = torch::nn::Linear(infeat,dmodel);
      embedy = torch::nn::Linear(1,dmodel);
      decoder = register_module("pfndec", torch::nn::Linear(dmodel, nbin));
    }

    torch::Tensor forward( const torch::Tensor& Xtrn,
                           const torch::Tensor& ytrn,
                           const torch::Tensor& Xtst )
    { 
      using namespace torch::indexing;
      auto train = embedx(Xtrn) + embedy(ytrn);
      auto test = embedx(Xtst);

      auto src = torch::cat({train,test},1);
      // I am doing this becase there is not batch first option here...
      src = src.permute({1, 0, 2});
      auto mask = att_mask(Xtrn.size(1)+Xtst.size(1), Xtst.size(1));
      return decoder.forward(encoder.forward(src, mask)).
        index({Slice(Xtrn.size(1), None), Slice(), Slice()});
    }

    // This is helper for creating the attention mask
    template<class O=double>
    torch::Tensor att_mask ( int size, int tstsize )
    {
      using namespace torch::indexing;
      int trnsize =  size - tstsize;
      auto mask = torch::zeros({size,size}) == 0.;
      mask.index({Slice(0,None),Slice(trnsize,None)}).zero_();
      mask |= torch::eye(size, mask.options()).eq(1);
      mask = mask.toType(torch::kFloat).masked_fill(mask==0,
                        -std::numeric_limits<float>::infinity());
      return mask.masked_fill(mask==1,0.);
    }

    int nparameters()
    {
      int total = 0;
      for (const auto& p : this->parameters())
      {
        total += p.numel();
      }
      return total;
    }

  };
}
