#ifndef PRINT_  
#define PRINT(x) std::cout << #x << " =\n" << x << std::endl;
#endif

// Macro to print variable name and value
#include <torch/torch.h>
#include <limits>
#include <iostream>
#include <typeinfo>
#include "utils.h"
#include "riemann.h"
#include "prior.h"
#include "model.h"

#ifndef PRINT_  
#define PRINT(x) std::cout << #x << " =\n" << x << std::endl;
#endif

// -----------------------------------------------------------
// Main
// -----------------------------------------------------------
int main(int argc, char** argv)
{
   CLIStore& cli = CLIStore::GetInstance();

  // -------------------------
  // Register flags
  // -------------------------
  cli.Register<int>("epochs", 10);                
  cli.Register<size_t>("seed", 25);              
  cli.Register<double>("lr", 0.001);           
  cli.Register<size_t>("bins", 100);           
  cli.Register<size_t>("nsamp", 100);           
  cli.Register<size_t>("nset", 20);           
  cli.Register<std::string>("mode", "predict");           
  cli.Register<std::string>("path", "./");           

  // -------------------------
  // Parse command line
  // -------------------------
  cli.Parse(argc, argv);

  // -------------------------
  // Access flag values
  // -------------------------
  auto epochs = cli.Get<int>("epochs");
  auto lr = cli.Get<double>("lr");
  auto seed = cli.Get<size_t>("seed");
  auto bins = cli.Get<size_t>("bins");
  auto nsamp = cli.Get<size_t>("nsamp");
  auto nset = cli.Get<size_t>("nset");

  // -------------------------
  // Print all registered flags
  // -------------------------
  cli.Print();

  torch::manual_seed(seed);


////////////////////////////////////////////////////////////////////////////////
/// There is something going wrong...
/// - Suspected places are
/// 3. I sample x differently and also do not normalize before embedding...->No
/// 4. Can also be the optimizer...->No
/// 5. Feeding noise with the test -> Potentially not
////////////////////////////////////////////////////////////////////////////////

/* //////////////////////////////////////////////////////////////////////////////// */
/* // TEST IDEA->Reimann Distribution */
/* //////////////////////////////////////////////////////////////////////////////// */
/*   auto borders = torch::arange(1,4).to(torch::kFloat); */
/*   dist::Riemann buck(borders,true); */
/*   PRINT(buck.forward(torch::zeros({1,1,2}),torch::ones({1,1,1}))) // == 0.693147 */
/* //////////////////////////////////////////////////////////////////////////////// */
  // Ahh limited memory... these big models sad sad sad 
  torch::Tensor borders;
  {
    auto res = prior::linear(100000, nsamp, 1);
    borders = dist::bin_borders(bins, c10::nullopt, std::get<1>(res));
  }


  model::SimplePFN model(256,4,4,512,1,2);
  PRINT(nparams(model));


  torch::optim::AdamW opt(model.parameters(),torch::optim::AdamWOptions(lr));
  dist::Riemann buck(borders,true);

  for (int epoch=0; epoch<=epochs; epoch++)
  {
    auto res = prior::linear(nset, nsamp, 1);
    auto sets = split(res,torch::randint(0,nsamp-1,1).item<int>());
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
    model.eval();
    std::cout << "\rEpoch ["
              << std::setw(3) << epoch << "/"
              << std::setw(3) << epochs << "] "
              << "Training Loss: " << std::setw(10) 
              << std::fixed << std::setprecision(6) << loss.item<float>() 
              << std::flush;
  }


  /* model.eval(); */
  /* auto res = prior::linear(1, nsamp, 1); */
  /* auto sets = split(res,10); */
  /* auto Xtrn = std::get<0>(sets); */ 
  /* auto Xtst = std::get<1>(sets); */
  /* auto ytrn = std::get<2>(sets); */
  /* auto ytst = std::get<3>(sets); */

  /* PRINT(buck.mean(model.forward(Xtrn,ytrn,Xtst))); */

////////////////////////////////////////////////////////////////////////////////  
  /* torch::Tensor x = torch::arange(0,5); */
  /* torch::Tensor x = torch::zeros({10,10}); */
  /* dist::Riemann buck(x); */
  /* PRINT(buck.forward(x,x)); */

  /* torch::Tensor ys = torch::rand(100); */
  /* auto borders = dist::bin_borders(5, c10::nullopt, ys); */
  /* PRINT(borders); */

  /* torch::Tensor lim = torch::tensor({1,100}); */
  /* PRINT(lim) */
  /* auto borders = dist::bin_borders(5, lim, c10::nullopt); */
  /* PRINT(borders); */

  /* auto res = prior::linear(2, 3, 2); */
  /* PRINT(std::get<0>(res)) */
  /* PRINT(std::get<1>(res)) */

  /* /1* torch::nn::Transformer model(torch::nn::TransformerOptions(512,8) *1/ */
  /* /1*                               .num_encoder_layers(6).num_decoder_layers(6)); *1/ */
  /* /1*                               .num_encoder_layers(6).num_decoder_layers(6)); *1/ */

  /* /1* PRINT(model::util::att_mask(5,2)); *1/ */ 

  /* model::SimplePFN model(512,8,6,6,1); */

  /* auto Xtrn = torch::randn({2,12,1}); */
  /* auto Xtst = torch::randn({2,8,1}); */
  /* auto ytrn = torch::randn({2,12,1}); */

  /* auto pred = model.forward(Xtrn, ytrn, Xtst); */

  /* auto src = torch::rand({10,32,512}); */
  /* torch::Tensor a, b; */
  /* auto idx = torch::randint(0,5,{2}); */
  /* /1* PRINT(model.encoder.forward(torch::ones({10,10,512}),a,b)); *1/ */
  /* PRINT(model.decoder.forward(torch::ones({10,10,512})).index_select(0,idx)); */
  /* model.encoder(src */
  /* auto tgt = torch::rand({20,32,512}); */
  /* PRINT(model(src,tgt)); */

  /* auto a = torch::rand({5}); */
  /* auto b = torch::rand({2,5}); */
  /* auto idx = torch::randint(0,5,{2}); */
  /* PRINT(idx) */
  /* PRINT(a); */
  /* PRINT(a.index_select(0,idx)); */
  /* PRINT(a.index_select(0,model::util::rest(idx,5))); */
  /* PRINT(b); */
  /* PRINT(b.index_select(1,idx)); */
  /* PRINT(b.index_select(1,model::util::rest(idx,5))); */


  return 0;
}
