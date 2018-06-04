#!/bin/bash

set -ev

###########################
# Get ComputeCpp
###########################
wget --no-check-certificate https://computecpp.codeplay.com/downloads/computecpp-ce/0.8.0/ubuntu-16.04-64bit.tar.gz
rm -rf /tmp/ComputeCpp && mkdir /tmp/ComputeCpp/
tar -xzf ubuntu-16.04-64bit.tar.gz -C /tmp/ComputeCpp --strip-components 1
ls -R /tmp/ComputeCpp/
