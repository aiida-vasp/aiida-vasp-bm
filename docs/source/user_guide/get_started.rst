===============
Getting started
===============

This page should contain a short guide on what the plugin does and
a short example on how to use the plugin.

Installation
++++++++++++

Use the following commands to install the plugin::

    git clone https://github.com/<yourgithubusername>/aiida-vasp-bm .
    cd aiida-vasp-bm
    pip install -e .  # also installs aiida, if missing (but not postgres)
    #pip install -e .[pre-commit,testing] # install extras for more features
    verdi quicksetup  # better to set up a new profile
    verdi plugin list aiida.calculations  # should now show your calculation plugins

Then use ``verdi code setup`` with the ``vasp_bm`` input plugin
to set up an AiiDA code for aiida-vasp-bm.

Usage
+++++

A quick demo of how to submit a calculation::

    verdi daemon start         # make sure the daemon is running
    cd examples
    verdi run test_submit.py        # submit test calculation
    verdi calculation list -a  # check status of calculation

If you have already set up your own aiida_vasp_bm code using
``verdi code setup``, you may want to try the following command::

    vasp_bm-submit  # uses aiida_vasp_bm.cli

Available calculations
++++++++++++++++++++++

.. aiida-calcjob:: DiffCalculation
    :module: aiida_vasp_bm.calculations
