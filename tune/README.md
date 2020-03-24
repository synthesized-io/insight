Installation
------------
    $  pip install ray
    

AWS Access
----------

Please do not forget to put pem file `~/.ssh/ray-autoscaler_eu-west-1.pem`


Usage
-----
Start/update cluster:

    $  ray up tune/tune-default.yaml
    

Submit a script:

    $  ./tune/submit.sh tune/sample_task.py 
    
Or access the jupyter notebook on the head node by forwarding port 8889 and visiting localhost:8889.

    $  ssh -L 8889:localhost:8889 -o IdentitiesOnly=yes -i ~/.ssh/ray-autoscaler_eu-west-1.pem ec2-user@--.--.--.--


Please note that you should always run this script from the root directory of synthesized. It's needed to sync sources.

