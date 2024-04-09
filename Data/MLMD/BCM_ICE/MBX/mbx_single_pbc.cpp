#include <cmath>
#include <cassert>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <cstdlib>

#include "io_tools/read_nrg.h"
#include "io_tools/write_nrg.h"

#include "bblock/system.h"

int main(int argc, char** argv) {

    /////////////////////////////
    // OPTION 1: Read NRG file //
    /////////////////////////////

    // Since C++ is being used, the built-in function ReadNrg can be used
    // Even if only one system is in the NRG file, declare a vector of systems
    std::vector<bblock::System> systems;

    // The json (settings) and nrg (monomers and coordinates) files
    std::string nrg = "input.nrg";
    std::string json = "mbx_pbc.json";

    // Convert the strings to char
    char nrg_c[nrg.size() + 1];
    std::strcpy(nrg_c,nrg.c_str());

    char json_c[json.size() + 1];
    std::strcpy(json_c,json.c_str());

    // Read the file and setup the system
    // Box and PBC options will be read from JSON
    tools::ReadNrg(nrg_c, systems);
    systems[0].SetUpFromJson(json_c);

    // Get the energy
    double en = systems[0].Energy(true);
    // Get the gradients
    std::vector<double> grads = systems[0].GetRealGrads();
    std::vector<double> chgs = systems[0].GetRealCharges();
    std::vector<double> vir = systems[0].GetVirial();
    std::vector<std::string> atn = systems[0].GetRealAtomNames();
    size_t n_atoms = systems[0].GetNumRealSites();
    // Print the energy
    std::string fileName = "mbx.out";
    std::ofstream writeFile (fileName.data());
    if (writeFile.is_open()) {
        writeFile << std::setprecision(10) << std::scientific << en << std::endl;
    
        for (size_t i = 0; i < n_atoms; i++) {
            double fx = -grads[3*i];
            double fy = -grads[3*i+1];
            double fz = -grads[3*i+2];
            double chg = chgs[i];

            writeFile << std::right << std::setprecision(8) << std::scientific << std::setw(20) 
                      << fx << std::setw(20) << fy << std::setw(20) << fz << std::setw(20) << chg << std::endl;
        }

        writeFile << std::right << std::setprecision(10) << std::scientific 
                  << std::setw(18) << vir[0] 
                  << std::setw(18) << vir[1]
                  << std::setw(18) << vir[2]
                  << std::setw(18) << vir[4] 
                  << std::setw(18) << vir[5]
                  << std::setw(18) << vir[8] << std::endl;
        writeFile.close();
    }

    
    return 0;
}
