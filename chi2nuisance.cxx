

// Pointer to your 2D function
TF2 *f2_generic = nullptr;

// 1D wrapper function evaluating min_y f(x,y)
double MinY_Wrapper(double *x, double *p) {
  double x_val = x[0];

  // Create a temporary 1D function along y for the current x
  // Limits [ymin, ymax] should match your TF2 range
  double ymin = f2_generic->GetYmin();
  double ymax = f2_generic->GetYmax();

  // Define lambda for f_x(y) = f(x_val, y)
  TF1 fy("fy", [x_val](double *y, double *) { return f2_generic->Eval(x_val, y[0]); }, ymin, ymax, 0);

  // Numerically find minimum along y
  return fy.GetMinimum(ymin, ymax);
}



void chi2nuisance(){

  float expected_yield = 12.5;
  int min = 0;
  int max = 30;

  //
  // Simulate: I measure N and I want to estimate "mu"
  //  exp (-mu) * mu^n/n!
  //

  //
  // Luminosity uncertainty 2%
  //

  float lumi_uncertainty = 0.02;
  // float lumi_uncertainty = 0.10;


  int N_measured = 14;

  // TF1 *f_likelihood = new TF1("f_likelihood", "TMath::Poisson([0], x)", min, max);
  TF2 *f_likelihood = new TF2("f_likelihood", "exp(-x*y) * TMath::Power(x*y, [0]) / TMath::Factorial([0]) * TMath::Gaus(y, 1.0, [1], true)", min, max, 0.5, 1.5);
  f_likelihood->SetParameter(0, N_measured);
  f_likelihood->SetParameter(1, lumi_uncertainty);
  f_likelihood->SetTitle("Likelihood;#mu;#theta;P(N,#mu)");

  TCanvas *c3_likelihood = new TCanvas("c3_likelihood", "Likelihood", 800, 600);
  f_likelihood->SetNpx(100);
  f_likelihood->SetNpy(100);
  f_likelihood->Draw("colz");

  c3_likelihood->SetGrid();




  TF2 *f_m2loglikelihood_not_shifted = new TF2("f_m2loglikelihood_not_shifted", "-2 * log (exp(-x*y) * TMath::Power(x*y, [0]) / TMath::Factorial([0]) * TMath::Gaus(y, 1.0, [1], true) )", min, max, 0.5, 1.5);
  f_m2loglikelihood_not_shifted->SetParameter(0, N_measured);
  f_m2loglikelihood_not_shifted->SetParameter(1, lumi_uncertainty);

  double x_min, y_min;
  f_m2loglikelihood_not_shifted->GetMinimumXY(x_min, y_min);
  float z_min = f_m2loglikelihood_not_shifted->Eval(x_min, y_min);

  std::cout << " x,y,z = " << x_min << " , " <<  y_min << " , " <<  z_min << std::endl;


  TF2 *f_m2loglikelihood = new TF2("f_m2loglikelihood", [f_m2loglikelihood_not_shifted, z_min](double *x, double *par) { return f_m2loglikelihood_not_shifted->Eval(x[0], x[1]) - z_min;}, min, max, 0.5, 1.5, 0);

  f_m2loglikelihood->SetTitle("-2 log Likelihood;#mu;#theta;-2 log P(N,#mu)");

  TCanvas *c3_m2loglikelihood = new TCanvas("c3_m2loglikelihood", "-2 log Likelihood", 800, 600);
  f_m2loglikelihood->SetNpx(100);
  f_m2loglikelihood->SetNpy(100);

  f_m2loglikelihood->SetContour(100);
  f_m2loglikelihood->SetMaximum(30.0);
  f_m2loglikelihood->SetMinimum(0.0);

  f_m2loglikelihood->Draw("colz");

  c3_m2loglikelihood->SetGrid();



  TCanvas *c3_m2loglikelihood_profile = new TCanvas("c3_m2loglikelihood_profile", "Likelihood profile", 800, 600);

  // f2_generic = (TF2*) f_m2loglikelihood_not_shifted->Clone();
  // f2_generic = (TF2*) f_m2loglikelihood->Clone();
  f2_generic = f_m2loglikelihood;

  // f2_generic->SetNpx(10);
  // f2_generic->SetNpy(10);

  // f2_generic->SetContour(100);
  // f2_generic->SetMaximum(30.0);
  // f2_generic->SetMinimum(0.0);


  // f2_generic->Draw("colz");

  TF1 *f_m2loglikelihood_profiled = new TF1("f_m2loglikelihood_profiled", MinY_Wrapper, f2_generic->GetXmin(), f2_generic->GetXmax(), 0);
  f_m2loglikelihood_profiled->SetTitle("-2 log Likelihood;#mu;P(N,#mu)");
  f_m2loglikelihood_profiled->SetLineColor(kRed);

  f_m2loglikelihood_profiled->SetMaximum(10);
  f_m2loglikelihood_profiled->Draw();



  TF1 *f_simple_m2LogLikelihood_not_shifted = new TF1("f_simple_m2LogLikelihood_not_shifted", "-2 * log (exp(-x) * TMath::Power(x, [0]) / TMath::Factorial([0]))", min, max);
  f_simple_m2LogLikelihood_not_shifted->SetParameter(0, N_measured);


  x_min = f_simple_m2LogLikelihood_not_shifted->GetMinimumX(min, max);
  y_min = f_simple_m2LogLikelihood_not_shifted->Eval(x_min);

  TF1 *f_simple_m2LogLikelihood = new TF1("f_delta", [f_simple_m2LogLikelihood_not_shifted, y_min](double *x, double *par) { return f_simple_m2LogLikelihood_not_shifted->Eval(x[0]) - y_min;}, min, max, 0);

  f_simple_m2LogLikelihood->SetTitle("-2 * Log Likelihood;#mu;-2 log P(N,#mu)");

  f_simple_m2LogLikelihood->SetLineColor(kBlue);

  f_simple_m2LogLikelihood->Draw("same");


  c3_m2loglikelihood_profile->SetGrid();


}



