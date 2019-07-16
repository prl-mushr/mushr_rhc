[![Build Status](https://dev.azure.com/prl-mushr/mushr_rhc/_apis/build/status/prl-mushr.mushr_rhc?branchName=master)](https://dev.azure.com/prl-mushr/mushr_rhc/_build/latest?definitionId=1&branchName=master)

# Receding Horizon Control

This module hosts the RHC controller first implemented on MuSHR stack. It is a model predictive contoller that plans to waypoints from a goal (instead of a reference trajectory). This controller is suitable for cars that don't have a planning module, but want simple MPC.

## Installing on the car
Get pip:
```
sudo apt install python-pip
```
To run this module on the car, you need a few packages. To get them download the wheel file for torch from nvidia:
```
$ https://nvidia.box.com/shared/static/m6vy0c7rs8t1alrt9dqf7yt1z587d1jk.whl torch-1.1.0a0+b457266-cp27-cp27mu-linux_aarch64.whl
$ pip install torch-1.1.0a0+b457266-cp27-cp27mu-linux_aarch64.whl
```
Then get the future package:
```
pip install future
```
Then get the python packages necessary:
```
$ sudo apt install python-scipy
$ sudo apt install python-networkx
$ sudo apt install python-sklearn
```
