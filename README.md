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

    ./analysis.sh



NB:

    sudo apt install meson ninja-build


