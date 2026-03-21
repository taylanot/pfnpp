#ifndef PRINT_  
#define PRINT(x) std::cout << #x << " =\n" << x << std::endl;
#endif

// Macro to print variable name and value
#include <torch/torch.h>
#include <filesystem>
#include <limits>
#include <iostream>
#include <typeinfo>
#include "utils.h"
/* const torch::Device DEVICE = select_device(); */
#include "riemann.h"
#include "prior.h"
#include "model.h"
#include "train.h"


#ifndef PRINT_  
#define PRINT(x) std::cout << #x << " =\n" << x << std::endl;
#endif

// -----------------------------------------------------------
// Main
// -----------------------------------------------------------
int main(int argc, char** argv)
{
  namespace fs = std::filesystem;
  CLIStore& conf = CLIStore::GetInstance();

  // -------------------------
  // Register flags
  // -------------------------
  conf.Register<int>("epochs", 10);                
  conf.Register<size_t>("seed", 25);              
  conf.Register<double>("lr", 0.001);           
  conf.Register<size_t>("nbin", 100);           
  conf.Register<size_t>("nsamp", 100);           
  conf.Register<size_t>("nfeat", 1);           
  conf.Register<size_t>("nset", 20);           
  conf.Register<size_t>("checks", 20);           
  conf.Register<fs::path>("path", "./simple");           

  // -------------------------
  // Parse command line
  // -------------------------
  conf.Parse(argc, argv);

  // -------------------------
  // Access flag values
  // -------------------------
  auto lr = conf.Get<double>("lr");
  auto seed = conf.Get<size_t>("seed");

  // -------------------------
  // Print all registered flags
  // -------------------------
  conf.Print();

  // -------------------------
  // Create the path
  // -------------------------
  torch::manual_seed(seed);


  auto pr = prior::LinearTasks(0, 1, 1);

  if (!is_regular_file(conf.Get<fs::path>("path")))
  {
    fs::create_directories(conf.Get<fs::path>("path"));
    model::SimplePFN pfn(pr, conf.Get<size_t>("nsamp"));
    torch::optim::AdamW opt(pfn->parameters(),torch::optim::AdamWOptions(lr));
    train::Simple(pr, pfn, opt, conf);
  }
  else
  {
    model::SimplePFN pfn(pr, conf.Get<size_t>("nsamp"));
    auto epoch = load_checkpoint(conf.Get<fs::path>("path"), pfn);
    conf.Set<fs::path>("path",conf.Get<fs::path>("path").remove_filename());
    torch::optim::AdamW opt(pfn->parameters(),torch::optim::AdamWOptions(lr));
    train::Simple( pr, pfn, opt, conf, epoch );
  }

  return 0;
}
