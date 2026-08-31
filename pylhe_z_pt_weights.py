import pylhe
import matplotlib.pyplot as plt

# Load an LHE file
lhe_data = pylhe.LHEFile.fromfile("SMEFT_Z_all_weights/Events/run_01/unweighted_events.lhe.gz")
events = lhe_data.events

# Change production to Z
# generate p p > z j j
#
# generate p p > l+ l- j j
#
# or reconstruct from 2 leptons rthe pt
#
# mll from 2 leptons too
#


Z_PDGID = 23

z_pt_values = []
z_weights = []

z_weights_sm = []
z_weights_chdd_m1 = []
z_weights_chdd_p1 = []

# Iterate over events
# for event in events:
#   weight = event.eventinfo.weight
#   for particle in event.particles:
#     if particle.id == Z_PDGID:
#       pt = (particle.px**2 + particle.py**2) ** 0.5
#       z_pt_values.append(pt)
#       z_weights.append(weight)
#       z_weights_sm      .append( event.weights["sm"]      )
#       z_weights_chdd_m1 .append( event.weights["chdd_m1"] )
#       z_weights_chdd_p1 .append( event.weights["chdd_p1"] )
#
# print(f"Found {len(z_pt_values)} Z bosons")
# print(f"Sum of weights: {sum(z_weights):.4f}")

#
# if no-Z found, go for 2-leptons
#

LEPTON_PDGIDS = {11, -11, 13, -13, 15, -15}  # e, mu, tau (either charge)

if len(z_pt_values) == 0 :
  for event in events:
    weight = event.eventinfo.weight

    leptons = [p for p in event.particles if p.id in LEPTON_PDGIDS and p.status == 1]
    px = 0
    py = 0
    pz = 0
    e  = 0
    for lep in leptons:
      px += lep.px
      py += lep.py
      pz += lep.pz
      e  += lep.e

    mass2 = e**2 - px**2 - py**2 - pz**2
    mass = mass2**0.5 if mass2 > 0 else 0.0  # guard against tiny negative values from rounding
    # z_pt_values.append( mass )
    z_pt_values.append( ( px*px + py*py ) ** 0.5)

    z_weights.append(weight)
    z_weights_sm      .append( event.weights["sm"]      )
    z_weights_chdd_m1 .append( event.weights["chdd_m1"] )
    z_weights_chdd_p1 .append( event.weights["chdd_p1"] )


print(f"Found {len(z_pt_values)} Z bosons")
print(f"Sum of weights: {sum(z_weights):.4f}")


# Plot the pT distribution, weighted

bins = 100  # keep the same binning for all 4, so shapes are directly comparable

plt.figure(figsize=(8, 6))
plt.hist(z_pt_values, bins=bins, weights=z_weights,          histtype="step", linewidth=1.5, color="black",  label="nominal (eventinfo.weight)")
plt.hist(z_pt_values, bins=bins, weights=z_weights_sm,       histtype="step", linewidth=1.5, color="tab:blue",  label="sm")
plt.hist(z_pt_values, bins=bins, weights=z_weights_chdd_m1,  histtype="step", linewidth=1.5, color="tab:red",   label="chdd_m1")
plt.hist(z_pt_values, bins=bins, weights=z_weights_chdd_p1,  histtype="step", linewidth=1.5, color="tab:green", label="chdd_p1")

plt.xlabel("Z boson $p_T$ [GeV]")
# plt.xlabel("Z boson Invariant mass [GeV]")
plt.ylabel("Weighted events")
plt.title("Z boson transverse momentum — weight comparison")
# plt.title("Z boson invariant mass — weight comparison")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("z_pt_weight_comparison.png", dpi=150)
plt.show()








