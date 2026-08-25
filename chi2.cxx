

void chi2(){

  float expected_yield = 12.5;
  int min = 0;
  int max = 30;

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
  f_m2LogLikelihood->SetTitle("-2 * Log Likelihood;#mu;-2 log P(N,#mu)");

  TCanvas *c4_m2LogLikelihood = new TCanvas("c4_m2LogLikelihood", "-2 LogLikelihood", 800, 600);
  f_m2LogLikelihood->Draw();


  float min_x_draw = 5;
  float max_x_draw = 25;

  double x_min = f_m2LogLikelihood->GetMinimumX(min, max);
  double y_min = f_m2LogLikelihood->Eval(x_min);

  TF1 *f_delta = new TF1("f_delta", [f_m2LogLikelihood, y_min](double *x, double *par) { return f_m2LogLikelihood->Eval(x[0]) - y_min;}, min, max, 0);

  TCanvas *c5_m2LogLikelihood_shifted = new TCanvas("c5_m2LogLikelihood_shifted", "-2 LogLikelihood shifted", 800, 600);

  f_delta->SetTitle("-2 * Log Likelihood;#mu;-2 log P(N,#mu)");
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
  // Build a chi2
  //

  // TF1 *f_chi2 = new TF1("f_chi2", "(x - [0])*(x - [0]) / [0]", min, max);
  TF1 *f_chi2 = new TF1("f_chi2", "(x - [0])*(x - [0]) / x", min, max);
  f_chi2->SetParameter(0, N_measured);
  f_chi2->SetTitle("#chi^{2};#mu;#chi^{2} (N,#mu)");
  f_chi2->SetLineColor(kBlue +1);

  TCanvas *c8_chi2 = new TCanvas("c8_chi2", "Chi2", 800, 600);
  f_chi2->Draw();

  f_chi2->GetXaxis()->SetRangeUser(min_x_draw, max_x_draw);

  llik_1->Draw();
  llik_4->Draw();

  x_min = f_chi2->GetMinimumX(min, max);
  y_min = f_chi2->Eval(x_min);
  mu_low  = f_chi2->GetX(1.0, min, x_min);
  mu_high = f_chi2->GetX(1.0, x_min, max);

  std::cout << "chi2" << std::endl;
  std::cout << "x_min = " << x_min << std::endl;
  std::cout << "y_min = " << y_min << std::endl;


  TLine *l_low_chi2 = new TLine(mu_low, 0, mu_low, 1.0);
  l_low_chi2->SetLineWidth(3);
  l_low_chi2->SetLineColor(kRed);
  l_low_chi2->SetLineStyle(3); // Dotted line
  l_low_chi2->Draw();

  TLine *l_high_chi2 = new TLine(mu_high, 0, mu_high, 1.0);
  l_high_chi2->SetLineWidth(3);
  l_high_chi2->SetLineColor(kRed);
  l_high_chi2->SetLineStyle(3);
  l_high_chi2->Draw();


  c8_chi2->SetGrid();


  TF1 *f_delta_chi2 = new TF1("f_delta_chi2", [f_chi2, y_min](double *x, double *par) { return f_chi2->Eval(x[0]) - y_min;}, min, max, 0);

  TCanvas *c8_chi2_shifted = new TCanvas("c8_chi2_shifted", "Chi2 shifted", 800, 600);

  f_delta_chi2->SetTitle("#chi^{2};#mu;#chi^{2} (N,#mu)");
  f_delta_chi2->SetLineColor(kBlue + 1);
  f_delta_chi2->SetLineWidth(2);
  f_delta_chi2->GetXaxis()->SetRangeUser(min_x_draw, max_x_draw);
  f_delta_chi2->Draw();

  TLine* chi2_1 = new TLine(min_x_draw, 1, max_x_draw, 1);
  chi2_1->SetLineWidth(2);
  chi2_1->SetLineColor(kRed);
  chi2_1->Draw();

  c8_chi2_shifted->SetGrid();


}



