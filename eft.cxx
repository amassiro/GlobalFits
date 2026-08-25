
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



void eft(){

  //
  // function
  //

  float expected_yield_SM   = 12.5;
  float expected_yield_Lin  = -1;
  float expected_yield_Quad =  0.1;

  int min = -20;
  int max = 40;

  // gStyle->SetPalette(kViridis);
  // gStyle->SetPalette(kBird);
  // gStyle->SetPalette(kCividis);
  gStyle->SetPalette(kCool);
  // gStyle->SetPalette(kTemperatureMap);


  int total_colors = gStyle->GetNumberOfColors();

  int N_measured = 14;



  //
  // -2 log Likelihood
  //


  TString mu_definition = Form ("%f + x * (%f) + x * x * (%f)", expected_yield_SM, expected_yield_Lin, expected_yield_Quad);
  std::cout << "mu_definition = " << mu_definition << std::endl;

  //                                                              N, mu
  TString function_definition = Form ("- 2 * log (TMath::Poisson([0],%s))", mu_definition.Data());
  std::cout << "function_definition = " << function_definition << std::endl;
  TF1 *f_m2LogLikelihood = new TF1("f_m2LogLikelihood", function_definition.Data(), min, max);
  f_m2LogLikelihood->SetNpx(100);

  f_m2LogLikelihood->SetParameter(0, N_measured);
  f_m2LogLikelihood->SetTitle("-2 * Log Likelihood;c;-2 log P(N,c)");

  TCanvas *c4_m2LogLikelihood = new TCanvas("c4_m2LogLikelihood", "-2 LogLikelihood", 800, 600);
  f_m2LogLikelihood->Draw();


  float min_x_draw = -10;
  float max_x_draw = 20;


  double x_min = f_m2LogLikelihood->GetMinimumX(min, max);
  double y_min = f_m2LogLikelihood->Eval(x_min);

  std::cout << "x_min = " << x_min << std::endl;
  std::cout << "y_min = " << y_min << std::endl;


  TF1 *f_delta = new TF1("f_delta", [f_m2LogLikelihood, y_min](double *x, double *par) { return f_m2LogLikelihood->Eval(x[0]) - y_min;}, min, max, 0);
  f_delta->SetNpx(100);

  TCanvas *c5_m2LogLikelihood_shifted = new TCanvas("c5_m2LogLikelihood_shifted", "-2 LogLikelihood shifted", 800, 600);

  f_delta->SetTitle("-2 * Log Likelihood;c;-2 log P(N,c)");
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

  function_definition = Form ("(%s - [0])*(%s - [0]) / (%s)" , mu_definition.Data(), mu_definition.Data(), mu_definition.Data());
  std::cout << "function_definition = " << function_definition << std::endl;


  TF1 *f_chi2 = new TF1("f_chi2", function_definition.Data(), min, max);
  f_chi2->SetParameter(0, N_measured);
  f_chi2->SetTitle("#chi^{2};c;#chi^{2} (N,c)");
  f_chi2->SetLineColor(kBlue +1);

  TCanvas *c8_chi2 = new TCanvas("c8_chi2", "Chi2", 800, 600);
  f_chi2->Draw();

  f_chi2->GetXaxis()->SetRangeUser(min_x_draw, max_x_draw);

  llik_1->Draw();
  llik_4->Draw();

  x_min = f_chi2->GetMinimumX(min, max);
  mu_low  = f_chi2->GetX(1.0, min, x_min);
  mu_high = f_chi2->GetX(1.0, x_min, max);

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




  //
  // 2D: two operators
  //


  float expected_yield_Lin_A  = -1;
  float expected_yield_Quad_A =  0.1;
  float expected_yield_Lin_B  = 0.3;
  float expected_yield_Quad_B =  0.05;
  float expected_yield_Int_AB =  0.01;


  mu_definition = Form ("%f + x * (%f) + x * x * (%f) + y * (%f) + y * y * (%f) + x * y * (%f)", expected_yield_SM, expected_yield_Lin_A, expected_yield_Quad_A, expected_yield_Lin_B, expected_yield_Quad_B, expected_yield_Int_AB);
  std::cout << "mu_definition = " << mu_definition << std::endl;

  function_definition = Form ("(%s - [0])*(%s - [0]) / (%s)" , mu_definition.Data(), mu_definition.Data(), mu_definition.Data());
  std::cout << "function_definition = " << function_definition << std::endl;

  float min_x = -20;  // min;
  float max_x =  20;  // max;
  float min_y = -20;  // min;
  float max_y =  20;  // max;

  TF2 *f_chi2_2D = new TF2("f_chi2_2D", function_definition.Data(), min_x, max_x, min_y, max_y);

  f_chi2_2D->SetNpx(100);
  f_chi2_2D->SetNpx(100);

  f_chi2_2D->SetParameter(0, N_measured);
  f_chi2_2D->SetTitle("#chi^{2};cA;cB;#chi^{2} (N,cA, cB)");
  f_chi2_2D->SetLineColor(kBlue +1);

  TCanvas *c9_chi2_2D = new TCanvas("c9_chi2_2D", "Chi2 cA cB", 800, 600);
  f_chi2_2D->SetContour(100);
  f_chi2_2D->SetMaximum(30.0);
  f_chi2_2D->DrawClone("colz");



  f_chi2_2D->GetMinimumXY(x_min, y_min);
  // x_min = f_chi2_2D->GetMinimumX(min_x, max_x);
  // y_min = f_chi2_2D->GetMinimumY(min_y, max_y);

  double z_min = f_chi2_2D->Eval(x_min, y_min);

  std::cout << " ---- " << std::endl;
  std::cout << "minimum [" << x_min << " , " << y_min << "] = " << z_min << std::endl;
  std::cout << " ---- " << std::endl;


  TF2 *f_chi2_2D_contour = (TF2*) f_chi2_2D->Clone("f_chi2_2D_contour");

  double levels[1] = {2.30};
  f_chi2_2D_contour->SetContour(1, levels);

  f_chi2_2D_contour->SetLineColor(kRed);
  f_chi2_2D_contour->SetLineWidth(3);
  f_chi2_2D_contour->SetLineStyle(1);

  f_chi2_2D_contour->DrawClone("cont3 same");





  float min_y_draw = -10;
  float max_y_draw = 20;

  min_x_draw = min_x;
  max_x_draw = max_x;

  min_y_draw = min_y;
  max_y_draw = max_y;

  f_chi2_2D->GetXaxis()->SetRangeUser(min_x_draw, max_x_draw);
  f_chi2_2D->GetYaxis()->SetRangeUser(min_y_draw, max_y_draw);

  f_chi2_2D->GetZaxis()->SetRangeUser(0, 10);

  TCanvas *c9_chi2_2D_cont = new TCanvas("c9_chi2_2D_cont", "Chi2 cA cB", 800, 600);
  f_chi2_2D->Draw("surf");

  f_chi2_2D->GetXaxis()->SetTitleOffset(1.7);
  f_chi2_2D->GetYaxis()->SetTitleOffset(1.7);
  f_chi2_2D->GetZaxis()->SetTitleOffset(1.5);








  //
  // Build a chi2 for 2-experiments/bins
  //


  float expected_yield_alpha_SM   = 12.5;
  float expected_yield_alpha_Lin  = -1;
  float expected_yield_alpha_Quad =  0.1;
  float N_measured_alpha = 14;

//   float expected_yield_beta_SM   = 6.5;
//   float expected_yield_beta_Lin  = -0.1;
//   float expected_yield_beta_Quad =  0.2;
//   float N_measured_beta = 4;
//
  float expected_yield_beta_SM   = 22.5;
  float expected_yield_beta_Lin  = 1.5;
  float expected_yield_beta_Quad =  0.1;
  float N_measured_beta = 19;


  min = -20;
  max = 20;

  TString mu_alpha_definition = Form ("%f + x * (%f) + x * x * (%f)", expected_yield_alpha_SM, expected_yield_alpha_Lin, expected_yield_alpha_Quad);
  std::cout << "mu_alpha_definition = " << mu_alpha_definition << std::endl;
  TString mu_beta_definition = Form ("%f + x * (%f) + x * x * (%f)", expected_yield_beta_SM, expected_yield_beta_Lin, expected_yield_beta_Quad);
  std::cout << "mu_beta_definition = " << mu_beta_definition << std::endl;

  function_definition = Form ("(%s - [0])*(%s - [0]) / (%s) + (%s - [1])*(%s - [1]) / (%s)" , mu_alpha_definition.Data(), mu_alpha_definition.Data(), mu_alpha_definition.Data(), mu_beta_definition.Data(), mu_beta_definition.Data(), mu_beta_definition.Data());
  std::cout << "function_definition = " << function_definition << std::endl;

  TF1 *f_chi2_alpha_beta = new TF1("f_chi2_alpha_beta", function_definition.Data(), min, max);
  f_chi2_alpha_beta->SetParameter(0, N_measured_alpha);
  f_chi2_alpha_beta->SetParameter(1, N_measured_beta);
  f_chi2_alpha_beta->SetTitle("#chi^{2};c;#chi^{2} (N,c)");
  f_chi2_alpha_beta->SetLineColor(kOrange +1);


  x_min = f_chi2_alpha_beta->GetMinimumX(min, max);
  y_min = f_chi2_alpha_beta->Eval(x_min);

  TF1 *f_chi2_alpha_beta_shifted = new TF1("f_chi2_alpha_beta_shifted", [f_chi2_alpha_beta, y_min](double *x, double *par) { return f_chi2_alpha_beta->Eval(x[0]) - y_min;}, min, max, 0);

  f_chi2_alpha_beta_shifted->SetTitle("#chi^{2};c;#chi^{2} (N,c)");
  f_chi2_alpha_beta_shifted->SetLineColor(kOrange +1);


  mu_definition = Form ("%f + x * (%f) + x * x * (%f)", expected_yield_alpha_SM, expected_yield_alpha_Lin, expected_yield_alpha_Quad);

  function_definition = Form ("(%s - [0])*(%s - [0]) / (%s)" , mu_definition.Data(), mu_definition.Data(), mu_definition.Data());
  std::cout << "function_definition = " << function_definition << std::endl;

  TF1 *f_chi2_alpha = new TF1("f_chi2_alpha", function_definition.Data(), min, max);
  f_chi2_alpha->SetParameter(0, N_measured_alpha);
  f_chi2_alpha->SetTitle("#chi^{2};c;#chi^{2} (N,c)");
  f_chi2_alpha->SetLineColor(kBlue +1);

  mu_definition = Form ("%f + x * (%f) + x * x * (%f)", expected_yield_beta_SM, expected_yield_beta_Lin, expected_yield_beta_Quad);

  function_definition = Form ("(%s - [0])*(%s - [0]) / (%s)" , mu_definition.Data(), mu_definition.Data(), mu_definition.Data());
  std::cout << "function_definition = " << function_definition << std::endl;

  TF1 *f_chi2_beta = new TF1("f_chi2_beta", function_definition.Data(), min, max);
  f_chi2_beta->SetParameter(0, N_measured_beta);
  f_chi2_beta->SetTitle("#chi^{2};c;#chi^{2} (N,c)");
  f_chi2_beta->SetLineColor(kTeal +1);



  TCanvas *c10_chi2_alpha_beta = new TCanvas("c10_chi2_alpha_beta", "Chi2 #alpha, #beta", 1200, 250);

  c10_chi2_alpha_beta->Divide (3,1);

  f_chi2_alpha_beta_shifted->SetNpx(100);
  f_chi2_alpha             ->SetNpx(100);
  f_chi2_beta              ->SetNpx(100);

  c10_chi2_alpha_beta->cd(1);

  f_chi2_alpha_beta_shifted->Draw();
  f_chi2_alpha->Draw("same");
  f_chi2_beta->Draw("same");

  f_chi2_alpha_beta_shifted->GetYaxis()->SetRangeUser(0,10);

  TLegend *leg = new TLegend(0.68, 0.45, 0.88, 0.88);
  leg->SetBorderSize(1);
  leg->SetFillColor(kWhite);
  leg->SetTextSize(0.06);

  leg->AddEntry(f_chi2_alpha,      "#alpha", "l");
  leg->AddEntry(f_chi2_beta,       "#beta", "l");
  leg->AddEntry(f_chi2_alpha_beta, "#alpha + #beta", "l");

  leg->Draw();
  gPad->SetGrid();


  c10_chi2_alpha_beta->cd(2);
  f_chi2_alpha->Draw();
  f_chi2_alpha->GetYaxis()->SetRangeUser(0,10);
  TLegend *leg_alpha = new TLegend(0.68, 0.45, 0.88, 0.88);
  leg_alpha->SetBorderSize(1);
  leg_alpha->SetFillColor(kWhite);
  leg_alpha->SetTextSize(0.06);
  leg_alpha->AddEntry(f_chi2_alpha,      "#alpha", "l");
  leg_alpha->Draw();
  gPad->SetGrid();

  c10_chi2_alpha_beta->cd(3);
  f_chi2_beta->Draw();
  f_chi2_beta->GetYaxis()->SetRangeUser(0,10);
  TLegend *leg_beta = new TLegend(0.68, 0.45, 0.88, 0.88);
  leg_beta->SetBorderSize(1);
  leg_beta->SetFillColor(kWhite);
  leg_beta->SetTextSize(0.06);
  leg_beta->AddEntry(f_chi2_beta,      "#beta", "l");
  leg_beta->Draw();
  gPad->SetGrid();





  //
  // 2D: two operators and two experiments
  //

  min_x = -20;  // min;
  max_x =  20;  // max;
  min_y = -20;  // min;
  max_y =  20;  // max;


  expected_yield_alpha_SM   = 12.5;
  float expected_yield_alpha_Lin_A  = -1;
  float expected_yield_alpha_Quad_A =  0.1;
  float expected_yield_alpha_Lin_B  = 0.3;
  float expected_yield_alpha_Quad_B =  0.05;
  float expected_yield_alpha_Int_AB =  0.01;
  N_measured_alpha = 14;

  expected_yield_beta_SM   = 22.5;
  float expected_yield_beta_Lin_A  = 1.5;
  float expected_yield_beta_Quad_A =  0.1;
  float expected_yield_beta_Lin_B  = -0.1;
  float expected_yield_beta_Quad_B =  0.03;
  float expected_yield_beta_Int_AB =  0.1;
  N_measured_beta = 18;


  // expected_yield_beta_SM   = 22.5;
  // float expected_yield_beta_Lin_A  = 1.5;
  // float expected_yield_beta_Quad_A =  0.1;
  // float expected_yield_beta_Lin_B  = -0.1;
  // float expected_yield_beta_Quad_B =  0.03;
  // float expected_yield_beta_Int_AB =  0.5;
  // N_measured_beta = 19;


  mu_alpha_definition = Form ("%f + x * (%f) + x * x * (%f) + y * (%f) + y * y * (%f) + x * y * (%f)", expected_yield_alpha_SM, expected_yield_alpha_Lin_A, expected_yield_alpha_Quad_A, expected_yield_alpha_Lin_B, expected_yield_alpha_Quad_B, expected_yield_alpha_Int_AB);
  mu_beta_definition = Form ("%f + x * (%f) + x * x * (%f) + y * (%f) + y * y * (%f) + x * y * (%f)", expected_yield_beta_SM, expected_yield_beta_Lin_A, expected_yield_beta_Quad_A, expected_yield_beta_Lin_B, expected_yield_beta_Quad_B, expected_yield_beta_Int_AB);

  function_definition = Form ("(%s - [0])*(%s - [0]) / (%s) + (%s - [1])*(%s - [1]) / (%s)" , mu_alpha_definition.Data(), mu_alpha_definition.Data(), mu_alpha_definition.Data(), mu_beta_definition.Data(), mu_beta_definition.Data(), mu_beta_definition.Data());
  std::cout << "function_definition = " << function_definition << std::endl;

  TF2 *f_alpha_beta_chi2_2D = new TF2("f_alpha_beta_chi2_2D", function_definition.Data(), min_x, max_x, min_y, max_y);

  function_definition = Form ("(%s - [0])*(%s - [0]) / (%s)" , mu_alpha_definition.Data(), mu_alpha_definition.Data(), mu_alpha_definition.Data());
  TF2 *f_alpha_chi2_2D = new TF2("f_alpha_chi2_2D", function_definition.Data(), min_x, max_x, min_y, max_y);
  f_alpha_chi2_2D->SetParameter(0, N_measured_alpha);
  f_alpha_chi2_2D->SetTitle("#chi^{2};cA;cB;#chi^{2} (N,cA, cB)");

  function_definition = Form ("(%s - [0])*(%s - [0]) / (%s)" , mu_beta_definition.Data(), mu_beta_definition.Data(), mu_beta_definition.Data());
  TF2 *f_beta_chi2_2D  = new TF2("f_beta_chi2_2D" , function_definition.Data(), min_x, max_x, min_y, max_y);
  f_beta_chi2_2D->SetParameter(0, N_measured_beta);
  f_beta_chi2_2D->SetTitle("#chi^{2};cA;cB;#chi^{2} (N,cA, cB)");

  f_alpha_beta_chi2_2D->SetNpx(100);
  f_alpha_beta_chi2_2D->SetNpx(100);

  f_alpha_beta_chi2_2D->SetParameter(0, N_measured_alpha);
  f_alpha_beta_chi2_2D->SetParameter(1, N_measured_beta);
  f_alpha_beta_chi2_2D->SetTitle("#chi^{2};cA;cB;#chi^{2} (N,cA, cB)");
  f_alpha_beta_chi2_2D->SetLineColor(kBlue +1);

  TCanvas *c11_alpha_beta_chi2_2D = new TCanvas("c11_alpha_beta_chi2_2D", "Chi2 cA cB", 800, 600);

  f_alpha_beta_chi2_2D->GetMinimumXY(x_min, y_min);
  z_min = f_alpha_beta_chi2_2D->Eval(x_min, y_min);

  // x_min = 0;
  // y_min = 0;
  // z_min = 0;

  std::cout << " ---- " << std::endl;
  std::cout << "minimum [" << x_min << " , " << y_min << "] = " << z_min << std::endl;
  std::cout << " ---- " << std::endl;

  TF2 *f_alpha_beta_chi2_2D_shifted = new TF2("f_alpha_beta_chi2_2D_shifted", [f_alpha_beta_chi2_2D, z_min](double *x, double *par) { return f_alpha_beta_chi2_2D->Eval(x[0], x[1]) - z_min;}, min_x, max_x, min_y, max_y, 0);

  f_alpha_beta_chi2_2D_shifted->SetTitle("#chi^{2};cA;cB;#chi^{2} (N,cA, cB)");
  f_alpha_beta_chi2_2D_shifted->SetContour(100);
  f_alpha_beta_chi2_2D_shifted->SetMaximum(30.0);
  f_alpha_beta_chi2_2D_shifted->DrawClone("colz");


  TF2 *f_alpha_beta_chi2_2D_shifted_contour = (TF2*) f_alpha_beta_chi2_2D_shifted->Clone("f_alpha_beta_chi2_2D_shifted_contour");

  // double levels[1] = {2.30};
  f_alpha_beta_chi2_2D_shifted_contour->SetContour(1, levels);

  f_alpha_beta_chi2_2D_shifted_contour->SetLineColor(kRed);
  f_alpha_beta_chi2_2D_shifted_contour->SetLineWidth(3);
  f_alpha_beta_chi2_2D_shifted_contour->SetLineStyle(1);

  f_alpha_beta_chi2_2D_shifted_contour->DrawClone("cont3 same");


  TCanvas *c11_alpha_chi2_2D = new TCanvas("c11_alpha_chi2_2D", "Chi2 cA cB", 800, 600);
  f_alpha_chi2_2D->GetMinimumXY(x_min, y_min);
  z_min = f_alpha_chi2_2D->Eval(x_min, y_min);

  TF2 *f_alpha_chi2_2D_shifted = new TF2("f_alpha_chi2_2D_shifted", [f_alpha_chi2_2D, z_min](double *x, double *par) { return f_alpha_chi2_2D->Eval(x[0], x[1]) - z_min;}, min_x, max_x, min_y, max_y, 0);

  f_alpha_chi2_2D_shifted->SetTitle("#chi^{2};cA;cB;#chi^{2} (N,cA, cB)");
  f_alpha_chi2_2D_shifted->SetContour(100);
  f_alpha_chi2_2D_shifted->SetMaximum(30.0);
  f_alpha_chi2_2D_shifted->DrawClone("colz");

  TF2 *f_alpha_chi2_2D_shifted_contour = (TF2*) f_alpha_chi2_2D_shifted->Clone("f_alpha_chi2_2D_shifted_contour");

  f_alpha_chi2_2D_shifted_contour->SetContour(1, levels);

  f_alpha_chi2_2D_shifted_contour->SetLineColor(kRed);
  f_alpha_chi2_2D_shifted_contour->SetLineWidth(3);
  f_alpha_chi2_2D_shifted_contour->SetLineStyle(1);

  f_alpha_chi2_2D_shifted_contour->DrawClone("cont3 same");






  TCanvas *c11_beta_chi2_2D = new TCanvas("c11_beta_chi2_2D", "Chi2 cA cB", 800, 600);
  f_beta_chi2_2D->GetMinimumXY(x_min, y_min);
  z_min = f_beta_chi2_2D->Eval(x_min, y_min);

  TF2 *f_beta_chi2_2D_shifted = new TF2("f_beta_chi2_2D_shifted", [f_beta_chi2_2D, z_min](double *x, double *par) { return f_beta_chi2_2D->Eval(x[0], x[1]) - z_min;}, min_x, max_x, min_y, max_y, 0);

  f_beta_chi2_2D_shifted->SetTitle("#chi^{2};cA;cB;#chi^{2} (N,cA, cB)");
  f_beta_chi2_2D_shifted->SetContour(100);
  f_beta_chi2_2D_shifted->SetMaximum(30.0);
  f_beta_chi2_2D_shifted->DrawClone("colz");

  TF2 *f_beta_chi2_2D_shifted_contour = (TF2*) f_beta_chi2_2D_shifted->Clone("f_beta_chi2_2D_shifted_contour");

  f_beta_chi2_2D_shifted_contour->SetContour(1, levels);

  f_beta_chi2_2D_shifted_contour->SetLineColor(kRed);
  f_beta_chi2_2D_shifted_contour->SetLineWidth(3);
  f_beta_chi2_2D_shifted_contour->SetLineStyle(1);

  f_beta_chi2_2D_shifted_contour->DrawClone("cont3 same");



  c11_alpha_beta_chi2_2D->cd();

  f_alpha_chi2_2D_shifted_contour->SetLineColor(kRed+4);
  f_beta_chi2_2D_shifted_contour->SetLineColor(kRed+4);
  f_alpha_chi2_2D_shifted_contour->SetLineStyle(2);
  f_beta_chi2_2D_shifted_contour->SetLineStyle(2);

  f_alpha_chi2_2D_shifted_contour->DrawClone("cont3 same");
  f_beta_chi2_2D_shifted_contour->DrawClone("cont3 same");



  // TF2 *f2_68 = (TF2*)f2_dchi2->Clone("f2_68");
  // double level_68[1] = {2.30};
  // f2_68->SetContour(1, level_68);
  // f2_68->SetLineColor(kBlack);
  // f2_68->SetLineWidth(2);
  // f2_68->Draw("cont3 same");
  //
  // TF2 *f2_95 = (TF2*)f2_dchi2->Clone("f2_95");
  // double level_95[1] = {5.99};
  // f2_95->SetContour(1, level_95);
  // f2_95->SetLineColor(kBlack);
  // f2_95->SetLineWidth(2);
  // f2_95->SetLineStyle(kDashed);
  // f2_95->Draw("cont3 same");
  //

  gPad->SetGrid();


  TCanvas *c12_profile_and_2D = new TCanvas("c12_profile_and_2D", "Chi2 cA cB", 1200, 600);

  c12_profile_and_2D->Divide(2,1);

  c12_profile_and_2D->cd(1);

  f_alpha_beta_chi2_2D_shifted->SetTitle("#chi^{2};cA;cB;#chi^{2} (N,cA, cB)");
  f_alpha_beta_chi2_2D_shifted->SetContour(100);
  f_alpha_beta_chi2_2D_shifted->SetMaximum(30.0);
  f_alpha_beta_chi2_2D_shifted->DrawClone("colz");

  TF2 *f_alpha_beta_chi2_2D_shifted_contour_many_contours = (TF2*) f_alpha_beta_chi2_2D_shifted->Clone("f_alpha_beta_chi2_2D_shifted_contour_many_contours");

  double many_levels[10] = {1,2,3,4,5,6,7,8,9,10};
  f_alpha_beta_chi2_2D_shifted_contour_many_contours->SetContour(10, many_levels);

  f_alpha_beta_chi2_2D_shifted_contour_many_contours->SetLineColor(kRed);
  f_alpha_beta_chi2_2D_shifted_contour_many_contours->SetLineWidth(3);
  f_alpha_beta_chi2_2D_shifted_contour_many_contours->SetLineStyle(1);

  f_alpha_beta_chi2_2D_shifted_contour_many_contours->DrawClone("cont3 same");


  c12_profile_and_2D->cd(2);

  f2_generic = (TF2*) f_alpha_beta_chi2_2D_shifted->Clone();
  TF1 *f_alpha_beta_chi2_2D_shifted_profiled = new TF1("f_alpha_beta_chi2_2D_shifted_profiled", MinY_Wrapper, f2_generic->GetXmin(), f2_generic->GetXmax(), 0);
  f_alpha_beta_chi2_2D_shifted_profiled->SetTitle("min_{cB} #chi2(cA,cB);cA;min_{cB} #chi2(cA,cB)");
  f_alpha_beta_chi2_2D_shifted_profiled->SetLineColor(kRed);

  f_alpha_beta_chi2_2D_shifted_profiled->Draw();


}



