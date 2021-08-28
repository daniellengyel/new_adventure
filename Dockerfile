
FROM quay.io/dolfinadjoint/pyadjoint:latest

USER root


RUN /bin/bash -l -c "pip3 install --no-cache --ignore-installed jax[cpu]"

# ADD /moola-master /moola-master
# RUN /bin/bash -l -c "pip3 uninstall moola"


USER fenics


RUN /bin/bash -l -c "python3 -c \"import fenics_adjoint\""
RUN /bin/bash -l -c "python3 -c \"import dolfin; import pyadjoint.ipopt\""

USER root
