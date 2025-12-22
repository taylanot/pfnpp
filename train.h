/*
  * Author: Ozgur Taylan Turan
  * Date: 18 December 2025
  * Description: This is for all the priors you might want to train on ...
  *
*/
#pragma once
#include<optional> 
#include<functional> 
#include <chrono>

namespace train 
{
  template<class PRIOR, class MODEL, class OPT, class DTYPE=float>
  void Simple ( const PRIOR& prior, MODEL& model, OPT& opt,
                const CLIStore& conf, const torch::Device device )
  {

    model->to(device);

    auto epochs = conf.Get<size_t>("epochs");

    auto t_total_start = std::chrono::high_resolution_clock::now();
    double cumulative_epoch_time = 0.0;

    for (size_t epoch = 0; epoch <= epochs; epoch++)
    {
      auto t_epoch_start = std::chrono::high_resolution_clock::now();

      auto res = prior.Sample(conf.Get<size_t>("nset"),
                              conf.Get<size_t>("nsamp"),
                              conf.Get<size_t>("nfeat"));

      auto sets = split( res,
        torch::randint(0, conf.Get<size_t>("nsamp") - 1, 1).item<int>() );

      auto Xtrn = std::get<0>(sets);
      auto Xtst = std::get<1>(sets);
      auto ytrn = std::get<2>(sets);
      auto ytst = std::get<3>(sets);

      Xtrn = Xtrn.to(device); ytrn = ytrn.to(device);
      Xtst = Xtst.to(device); ytst = ytst.to(device);

      model->train();
      opt.zero_grad();

      auto loss = model( Xtrn,ytrn,Xtst,ytst );

      loss.backward();
      opt.step();

      auto t_epoch_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> epoch_time = t_epoch_end - t_epoch_start;

      cumulative_epoch_time += epoch_time.count();

      auto avg_epoch_time =
        cumulative_epoch_time / static_cast<DTYPE>(epoch + 1);

      auto remaining_time =
        avg_epoch_time * static_cast<DTYPE>(epochs - epoch);

      std::cout << "\rEpoch ["
                << std::setw(3) << epoch << "/"
                << std::setw(3) << epochs << "] "
                << "Loss: " << std::setw(10)
                << std::fixed << std::setprecision(6)
                << loss.template item<DTYPE>()
                << "  | Epoch: "
                << std::setw(6) << std::setprecision(3)
                << epoch_time.count() << " s"
                << "  ETA: "
                << format_time_dhms(remaining_time)
                << std::flush;
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<DTYPE> total_time = t_total_end - t_total_start;

    std::cout << "\nTotal training time: "
              << format_time_dhms(total_time.count())
              << "\n";

  }
}


