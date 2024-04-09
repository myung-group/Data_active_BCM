import json 
import sys      
import active

if __name__ == '__main__':
    
    fname_json = 'input.json'
    if len(sys.argv) > 1:
        fname_json = sys.argv[1]
            
    with open (fname_json) as f:
        json_data = json.load (f)

        
        main_calc = None
        calc = active.ActiveCalculator(json_data=json_data,
                                       calculator=main_calc)

        # change the trajectory filename 
        print ('json_data_fname_dat', json_data['fname_dat'])
        calc.include_data (json_data['traj'], json_data['fname_dat'])
       
        
