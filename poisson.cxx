

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

  int nEvents = 1000;

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

  // TF1 *f_likelihood = new TF1("f_likelihood", "TMath::Poisson([0], x)", min, max);
  TF1 *f_likelihood = new TF1("f_likelihood", "exp(-x) * TMath::Power(x, [0]) / TMath::Factorial([0])", min, max);
  f_likelihood->SetParameter(0, N_measured);
  f_likelihood->SetTitle("Likelihood;#mu;P(N,#mu)");

  TCanvas *c3_likelihood = new TCanvas("c3_likelihood", "Likelihood", 800, 600);
  f_likelihood->Draw();




  // TF1 *f_likelihood = new TF1("f_likelihood", "TMath::Poisson([0], x)", min, max);
  TF1 *f_m2LogLikelihood = new TF1("f_m2LogLikelihood", "-2 * log (exp(-x) * TMath::Power(x, [0]) / TMath::Factorial([0]))", min, max);
  f_m2LogLikelihood->SetParameter(0, N_measured);
  f_m2LogLikelihood->SetTitle("-2 * Log Likelihood;#mu;P(N,#mu)");

  TCanvas *c4_m2LogLikelihood = new TCanvas("c4_m2LogLikelihood", "-2 LogLikelihood", 800, 600);
  f_m2LogLikelihood->Draw();


  float min_x_draw = 5;
  float max_x_draw = 25;


  double x_min = f_m2LogLikelihood->GetMinimumX(min, max);
  double y_min = f_m2LogLikelihood->Eval(x_min);

  TF1 *f_delta = new TF1("f_delta", [f_m2LogLikelihood, y_min](double *x, double *par) { return f_m2LogLikelihood->Eval(x[0]) - y_min;}, min, max, 0);

  TCanvas *c5_m2LogLikelihood_shifted = new TCanvas("c5_m2LogLikelihood_shifted", "-2 LogLikelihood shifted", 800, 600);

  f_delta->SetTitle("-2 * Log Likelihood;#mu;P(N,#mu)");
  f_delta->SetLineColor(kBlue + 1);
  f_delta->SetLineWidth(2);
  f_delta->Draw();

  TLine* llik_1 = new TLine(min_x_draw, 1, max_x_draw, 1);
  llik_1->SetLineWidth(2);
  llik_1->SetLineColor(kRed);
  llik_1->Draw();

  TLine* llik_4 = new TLine(min_x_draw, 4, max_x_draw, 4);
  llik_4->SetLineWidth(2);
  llik_4->SetLineColor(kRed);
  llik_4->Draw();


  x_min = f_delta->GetMinimumX(min, max);
  float mu_low  = f_delta->GetX(1.0, min, x_min);
  float mu_high = f_delta->GetX(1.0, x_min, max);

  TLine *l_low = new TLine(mu_low, 0, mu_low, 1.0);
  l_low->SetLineWidth(3);
  l_low->SetLineColor(kRed);
  l_low->SetLineStyle(3); // Dotted line
  l_low->Draw();

  TLine *l_high = new TLine(mu_high, 0, mu_high, 1.0);
  l_high->SetLineWidth(3);
  l_high->SetLineColor(kRed);
  l_high->SetLineStyle(3);
  l_high->Draw();


  f_delta->GetXaxis()->SetRangeUser(min_x_draw, max_x_draw);

  c5_m2LogLikelihood_shifted->SetGrid();





  //
  // what is means 68% confidence level
  //

  int n_toys = 100;

  // negative log likelihood
  TF1 *f_nll = new TF1("f_likelihood", "-2 * TMath::Log(TMath::Poisson([0], x))", min, max);


  TH1F *h_mu_best  = new TH1F("h_mu_best", "Best Fit #mu;#mu;Toys", nbins*10, min-0.5, max-0.5);
  TH1F *h_mu_low   = new TH1F("h_mu_low",  "Lower 1#sigma Limit;#mu_{low};Toys",  nbins*10, min-0.5, max-0.5);
  TH1F *h_mu_high  = new TH1F("h_mu_high", "Upper 1#sigma Limit;#mu_{high};Toys", nbins*10, min-0.5, max-0.5);

  TH2F * h_support = new TH2F("h_support", ";#mu_{high};Toys", nbins*10, min-0.5, max-0.5, n_toys, 0, n_toys);

  float delta_xx = 1. * (max-min) / (nbins*10);
  std::cout << " delta_xx = " << delta_xx << std::endl;

  for (int itoy = 0; itoy<n_toys; itoy++) {

    int event_toy = rng.Poisson(expected_yield);

    // get confidence belt

    f_nll->SetParameter(0, event_toy);

    x_min = f_nll->GetMinimumX(min, max);
    y_min = f_nll->Eval(x_min);

    float target_y = y_min + 1.0;

    float min_x = (x_min > min) ? f_nll->GetX(target_y, min, x_min) : 0.0;
    float max_x = f_nll->GetX(target_y, x_min, max);

    h_mu_best->Fill(x_min);

    h_mu_low ->Fill(min_x);
    h_mu_high->Fill(max_x);

    int num_bins_active = (max_x-min_x)/(max-min)*nbins*10;
    for (int ii=0; ii<num_bins_active; ii++) {
      h_support->Fill(min_x+(ii+1)*delta_xx, itoy+1);
    }

    h_support->Fill(min_x, itoy+1);
    h_support->Fill(max_x, itoy+1);

    h_support->Fill(min_x, itoy+1);
    h_support->Fill(max_x, itoy+1);

  }

  TCanvas *c6_toys_belt = new TCanvas("c6_toys_belt", "Toys confidence belt", 800, 600);

  // h_mu_best->SetLineColor(kAzure + 2);
  // h_mu_low->SetLineColor(kTeal + 2);
  // h_mu_high->SetLineColor(kPink + 7);

  // h_mu_best->Draw("HIST");
  // h_mu_low->Draw("HIST SAME");
  // h_mu_high->Draw("HIST SAME");

  h_support->Draw("colz");

  gPad->SetGrid();








  //
  // Confidence Belt
  //

  int nbins_mu = nbins*40;
  TH2F * h_belt = new TH2F("h_belt", ";N;#mu", nbins, min, max, nbins_mu, min, max);

  float delta_yy = 1. * (max-min) / (nbins_mu+1);
  for (int ibins_mu = 0; ibins_mu<nbins_mu; ibins_mu++) {

    float mu_temporary = min + delta_yy*ibins_mu;
    f_poisson->SetParameter(0, mu_temporary);

    for (int ii = 0; ii<nbins; ii++) {
      h_belt->Fill(ii,mu_temporary,f_poisson->Eval(ii));
    }

  }

  TCanvas *c7_belt = new TCanvas("c7_belt", "Confidence belt", 800, 600);

  h_belt->Draw("colz");
  h_belt->GetXaxis()->SetTitle("N");
  h_belt->GetYaxis()->SetTitle("#mu");

  gPad->SetGrid();




}


