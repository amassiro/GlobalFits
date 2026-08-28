

void chi2comparison(){

  float expected_yield = 12.5;
  int min = 0;
  int max = 30;

  //
  // Simulate: I measure N and I want to estimate "mu"
  //  exp (-mu) * mu^n/n!
  //

  int N_measured = 14;

  float min_x_draw = 5;
  float max_x_draw = 25;

  double x_min = 0;
  double y_min = 0;


  //
  // Build a chi2
  //

  // TF1 *f_chi2 = new TF1("f_chi2", "(x - [0])*(x - [0]) / [0]", min, max);
  TF1 *f_chi2 = new TF1("f_chi2", "(x - [0])*(x - [0]) / x", min, max);
  f_chi2->SetParameter(0, N_measured);
  f_chi2->SetTitle("#chi^{2};#mu;#chi^{2} (N,#mu)");
  f_chi2->SetLineColor(kBlue +1);

  TF1 *f_chi2_alternative = new TF1("f_chi2_alternative", "(x - [0])*(x - [0]) / [0]", min, max);
  f_chi2_alternative->SetParameter(0, N_measured);
  f_chi2_alternative->SetTitle("#chi^{2};#mu;#chi^{2} (N,#mu)");
  f_chi2_alternative->SetLineColor(kRed +1);


  TCanvas *c8_chi2 = new TCanvas("c8_chi2", "Chi2", 800, 600);
  f_chi2->Draw();

  f_chi2->GetXaxis()->SetRangeUser(min_x_draw, max_x_draw);

  x_min = f_chi2->GetMinimumX(min, max);
  y_min = f_chi2->Eval(x_min);

  std::cout << "chi2" << std::endl;
  std::cout << "x_min = " << x_min << std::endl;
  std::cout << "y_min = " << y_min << std::endl;


  TF1 *f_delta_chi2 = new TF1("f_delta_chi2", [f_chi2, y_min](double *x, double *par) { return f_chi2->Eval(x[0]) - y_min;}, min, max, 0);


  x_min = f_chi2_alternative->GetMinimumX(min, max);
  y_min = f_chi2_alternative->Eval(x_min);

  std::cout << "chi2_alternative" << std::endl;
  std::cout << "x_min = " << x_min << std::endl;
  std::cout << "y_min = " << y_min << std::endl;

  f_chi2_alternative->Draw("same");


  TLegend *leg = new TLegend(0.38, 0.45, 0.78, 0.88);
  leg->SetBorderSize(1);
  leg->SetFillColor(kWhite);
  leg->SetTextSize(0.06);

  leg->AddEntry(f_chi2,  "#chi^{2} = (N-#mu)^{2}/#mu", "l");
  leg->AddEntry(f_chi2_alternative,  "#chi^{2} = (N-#mu)^{2}/N", "l");

  leg->Draw();

  c8_chi2->SetGrid();


  TF1 *f_delta_chi2_alternative = new TF1("f_delta_chi2_alternative", [f_chi2_alternative, y_min](double *x, double *par) { return f_chi2_alternative->Eval(x[0]) - y_min;}, min, max, 0);




  TCanvas *c8_chi2_shifted = new TCanvas("c8_chi2_shifted", "Chi2 shifted", 800, 600);

  f_delta_chi2->SetTitle("#chi^{2};#mu;#chi^{2} (N,#mu)");
  f_delta_chi2->SetLineColor(kBlue + 1);
  f_delta_chi2->SetLineWidth(2);
  f_delta_chi2->GetXaxis()->SetRangeUser(min_x_draw, max_x_draw);
  f_delta_chi2->Draw();

  f_delta_chi2_alternative->SetLineColor(kRed + 1);
  f_delta_chi2_alternative->SetLineWidth(2);
  f_delta_chi2_alternative->GetXaxis()->SetRangeUser(min_x_draw, max_x_draw);
  f_delta_chi2_alternative->Draw("same");

  TLine* chi2_1 = new TLine(min_x_draw, 1, max_x_draw, 1);
  chi2_1->SetLineWidth(2);
  chi2_1->SetLineColor(kRed);
  chi2_1->Draw();


  leg->Draw();


  c8_chi2_shifted->SetGrid();


}



