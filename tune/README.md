Installation
------------
    $  pip install ray 

Usage
-----

    $  ./tune/submit.sh tune/sample_task.py 

Please note that you should always run this script from the root directory of synthesized. It's needed to sync sources.

AWS Access
----------

Please do not forget to put pem file `~/.ssh/ray-autoscaler_eu-west-1.pem`


Advanced
--------

Update cluster:
    $  ray up tune/tune-default.yaml
