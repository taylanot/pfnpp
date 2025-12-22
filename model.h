/*
  * Author: Ozgur Taylan Turan
  * Date: 16 December 2025
  * Description: Collect the models here.
  *
*/

/* #include "riemann.h" */

namespace model
{
  struct SimplePFNImpl : torch::nn::Module
  {
    int dmodel_, nhead_, nencoder_, nhid_, infeat_, nbin_, nsamp_;
    prior::Tasks& pri_;

    torch::nn::TransformerEncoder encoder{nullptr};
    torch::nn::LayerNorm ln_between{nullptr};
    torch::nn::Linear decoder{nullptr}, embedx{nullptr}, embedy{nullptr};
    dist::Riemann loss = nullptr;

       SimplePFNImpl( prior::Tasks& pri,
                      const int nsamp, 
                      int dmodel=256,
                      int nhead=4,
                      int nencoder=4,
                      int nhid=512,
                      int infeat=1,
                      int nbin = 100 ) :  dmodel_(dmodel),
                                          nhead_(nhead),
                                          nencoder_(nencoder),
                                          nhid_(nhid),
                                          infeat_(infeat),
                                          nbin_(nbin),
                                          nsamp_(nsamp),
                                          pri_(pri)
          
    {
      // Encoder layer
      auto encoder_layer = torch::nn::TransformerEncoderLayer(
          torch::nn::TransformerEncoderLayerOptions(dmodel, nhead)
              .dim_feedforward(nhid)
              .dropout(0.));

      // Transformer encoder
      encoder = register_module("encoder", torch::nn::TransformerEncoder(
          torch::nn::TransformerEncoderOptions(encoder_layer, nencoder)
      ));

      // LayerNorm between encoder and decoder
      ln_between = register_module("ln_between", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({dmodel})));

      // Decoder
      decoder = register_module("decoder", torch::nn::Linear(dmodel,nbin));
      embedx = register_module("ex",torch::nn::Linear(infeat,dmodel));
      embedy = register_module("ey",torch::nn::Linear(1,dmodel));
      loss = register_module("loss", dist::Riemann(
                                              pri_.Border(nsamp_,infeat,nbin)));

      std::cout << "SimplePFN parameter count: " << nparams(*this) << std::endl;
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

    torch::Tensor forward( const torch::Tensor& Xtrn,
                           const torch::Tensor& ytrn,
                           const torch::Tensor& Xtst,
                           const c10::optional<torch::Tensor>& ytst )
    { 
      using namespace torch::indexing;
      auto train = embedx(Xtrn) + embedy(ytrn);
      auto test = embedx(Xtst);
      
      auto src = torch::cat({train,test},0);
      // I am doing this becase there is not batch first option here...
      /* src = src.permute({1, 0, 2}); */
      auto mask = att_mask(Xtrn.size(0)+Xtst.size(0), Xtst.size(0));
      mask = mask.to(DEVICE);
      if (ytst.has_value())
        return loss(decoder(encoder(src, mask)).
          index({Slice(Xtrn.size(0), None), Slice(), Slice()}),ytst.value());
      else
        return loss->mean(decoder(encoder(src, mask)).
          index({Slice(Xtrn.size(0), None), Slice(), Slice()}));
    }

  };

  TORCH_MODULE(SimplePFN);

}
