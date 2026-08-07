Global fits lectures and hands on
====

Developed for https://indico.cern.ch/event/1634357/


Simple poisson:

    r99t poisson.cxx


Simple EFT:

    r99t eft.cxx

Plain Python version:


    python3 -m venv myGlobalFit

    source myGlobalFit/bin/activate

    pip install matplotlib numpy scipy

    python3 eft.py

    python3 poisson.py


    deactivate


MG and SMEFT

    ./setup.sh

    ./create.sh

or

    ./create.sh fixed

Analysis:

    ./analysis.sh


NB:

    sudo apt install meson ninja-build




Manual:

    gunzip -k  MG5_aMC_v3_7_2/PROC_cHWB_linear/Events/run_01/unweighted_events.lhe.gz
    gunzip -k  MG5_aMC_v3_7_2/PROC_cHWB_quadratic/Events/run_01/unweighted_events.lhe.gz



    import model SMEFTsim_U35_MwScheme_UFO
    generate p p > z j j        NP=1 NP^2==1
    output SMEFT_Z_lin -f
    launch SMEFT_Z_lin


    generate u u~ > z j j        NP=1 NP^2==1



pylhe


    source myGlobalFit/bin/activate

    pip install pylhe

    python3 pylhe_example.py




