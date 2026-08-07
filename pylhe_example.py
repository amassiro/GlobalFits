import pylhe

# Load an LHE file
# lhe_data = pylhe.read_lhe("events.lhe")
lhe_data = pylhe.LHEFile.fromfile("SMEFT_Z_lin/Events/run_01/unweighted_events.lhe.gz")
events = lhe_data.events

# Iterate over events
for event in events:
    print("Event weight:", event.eventinfo.weight)
    print("Number of particles:", len(event.particles))

    for particle in event.particles:
        print(
            f"  PDG ID: {particle.id}, "
            f"Status: {particle.status}, "
            f"Px: {particle.px}, Py: {particle.py}, Pz: {particle.pz}, "
            f"Energy: {particle.e}"
        )
    print ("-----")
    # break  # just show the first event


