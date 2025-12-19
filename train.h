/*
  * Author: Ozgur Taylan Turan
  * Date: 18 December 2025
  * Description: This is for all the priors you might want to train on ...
  *
*/
#pragma once
#include<optional> 
#include<functional> 

namespace train 
{
  template<class PRIOR, class MODEL, class OPT>
  void RiemannLoss(const PRIOR& prior, MODEL& model, OPT& opt, const CLIStore& conf)
  {
    auto epochs = conf.Get<size_t>("epochs");
    torch::Tensor borders;
    {
      auto res = prior.Sample(100000, conf.Get<size_t>("nsamp"), 1);
      borders = dist::bin_borders(conf.Get<size_t>("nbin"),
                                  c10::nullopt, std::get<1>(res));
    }

    dist::Riemann buck(borders,true);

    for (size_t epoch=0; epoch<=epochs; epoch++)
    {
      auto res = prior.Sample(conf.Get<size_t>("nset"),
                              conf.Get<size_t>("nsamp"),
                              conf.Get<size_t>("nfeat"));
      auto sets = split(res,torch::randint(0,conf.Get<size_t>("nsamp")-1,1)
                                                                 .item<int>());
      // does not matter I guess...
      auto Xtrn = std::get<0>(sets); 
      auto Xtst = std::get<1>(sets);
      auto ytrn = std::get<2>(sets);
      auto ytst = std::get<3>(sets);
      model.train();
      opt.zero_grad();
      auto pred = model.forward(Xtrn, ytrn, Xtst);
      auto loss = buck.forward(pred,ytst);
      loss.backward();
      opt.step();
      // You can add a validation part as well, but for now it is extra work...
      std::cout << "\rEpoch ["

                << std::setw(3) << epoch << "/"
                << std::setw(3) << epochs << "] "
                << "Training Loss: " << std::setw(10) 
                << std::fixed << std::setprecision(6) << loss.template item<float>() 
                << std::flush;
    }
  }
}


