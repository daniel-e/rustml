#!/bin/bash

set -e

cd /root
wget -q https://raw.githubusercontent.com/daniel-e/rustml/master/build/run.sh
chmod +x run.sh
./run.sh
