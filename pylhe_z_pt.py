import pylhe
import matplotlib.pyplot as plt

# Load an LHE file
lhe_data = pylhe.LHEFile.fromfile("SMEFT_Z_lin/Events/run_01/unweighted_events.lhe.gz")
events = lhe_data.events

Z_PDGID = 23

z_pt_values = []
z_weights = []

# Iterate over events
for event in events:
    weight = event.eventinfo.weight
    for particle in event.particles:
        if particle.id == Z_PDGID:
            pt = (particle.px**2 + particle.py**2) ** 0.5
            z_pt_values.append(pt)
            z_weights.append(weight)

print(f"Found {len(z_pt_values)} Z bosons")
print(f"Sum of weights: {sum(z_weights):.4f}")

# Plot the pT distribution, weighted
plt.figure(figsize=(8, 6))
plt.hist(z_pt_values, bins=50, weights=z_weights, histtype="step", linewidth=1.5)
plt.xlabel("Z boson $p_T$ [GeV]")
plt.ylabel("Weighted events")
plt.title("Z boson transverse momentum")
# plt.yscale("log")
plt.tight_layout()
plt.savefig("z_pt.png", dpi=150)
plt.show()


PT_CUT = 200

# Weighted counts below/above the pT cut
weight_below = sum(w for pt, w in zip(z_pt_values, z_weights) if pt < PT_CUT)
weight_above = sum(w for pt, w in zip(z_pt_values, z_weights) if pt >= PT_CUT)

n_below = sum(1 for pt in z_pt_values if pt < PT_CUT)
n_above = sum(1 for pt in z_pt_values if pt >= PT_CUT)

print(f"pT < {PT_CUT} GeV:  {n_below} events, weighted sum = {weight_below:.4f}")
print(f"pT >= {PT_CUT} GeV: {n_above} events, weighted sum = {weight_above:.4f}")








