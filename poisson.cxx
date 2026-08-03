

void poisson(){

  //
  // function
  //

  float expected_yield = 12.5;
  int min = 0;
  int max = 50;

  TF1 *f_poisson = new TF1("f_poisson", "TMath::Poisson(x, [0])", min, max);

  f_poisson->SetParameter(0, expected_yield);
  f_poisson->SetTitle("Poisson Distribution;N;P(N)");

  TCanvas *c1_function = new TCanvas("c1_function", "Poisson Function", 800, 600);
  f_poisson->Draw();



  //
  // generate events
  //

  int nEvents = 10000;

  int nbins = max-min;

  TH1F *h_poisson = new TH1F("h_poisson", "Sampled Poisson;n;Entries", nbins, min-0.5, max-0.5);

  TRandom3 rng(0);

  float average_events = 0;
  for (int i = 0; i < nEvents; i++) {
    int event = rng.Poisson(expected_yield);
    average_events += event;
    h_poisson->Fill(event);
  }
  average_events /= nEvents;


  std::cout << "expected_yield = " << expected_yield << std::endl;
  std::cout << "average toys = " << average_events << std::endl;


  TCanvas *c2_generate_events = new TCanvas("c2_generate_events", "Poisson Histogram", 800, 600);
  h_poisson->SetFillColor(kBlue - 9);
  h_poisson->SetLineColor(kBlue + 2);
  h_poisson->Draw("HIST");





  //
  // Simulate: I measure N and I want to estimate "mu"
  //  exp (-mu) * mu^n/n!
  //

  int N_measured = 14;










}


