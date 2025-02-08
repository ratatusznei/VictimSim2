import sys
import os
import time

## importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer
from pathlib import Path

def main(data_folder_name, cfg_ag_folder):
      
    # Instantiate the environment
    env = Env(data_folder)
    
    # Instantiate agents rescuer and explorer
    rescuer_file = os.path.join(cfg_ag_folder, "rescuer_1_config.txt")
    master_rescuer = Rescuer(env, rescuer_file, master=True)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    for exp in range(1, 5):
        filename = f"explorer_{exp:1d}_config.txt"
        explorer_file = os.path.join(cfg_ag_folder, filename)
        Explorer(env, explorer_file, master_rescuer)
    

    # Run the environment simulator
    env.run()
    
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
        cfg_ag_folder = sys.argv[2]
    else:
        cur_folder = Path.cwd()

        data_folder = os.path.join(cur_folder.parent, "datasets", "data_408v_94x94")
        cfg_ag_folder = os.path.join(cur_folder, "cfg_1")
        
    main(data_folder, cfg_ag_folder)
