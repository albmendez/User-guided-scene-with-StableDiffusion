# **User-guided framework for scene generation using diffusion models**
<p align="center">
  <img src="./Pruebas_buenas/1_N8_0.png" height=200 />
</p>

This project presents a user-guided framework for scene generation using diffusion models. The proposed framework comprises a perception stage, followed by a solution generation, and a final-stage selection algorithm. With this architecture the robot reach a reasoning able to imagine new situations with and without context, depending on the input information given. To achieve the best relation between performance and execution time, different diffusion models were systematically evaluated under a zero-shot configuration. Also, we make tests to prove the accuracy of the method and how useful users find this application. These were made for various scenes and different applications like rearrangement objects, setting the table for dinner or ordering a messy desktop.

# Installation
To be used on your device, follow the installation steps below.

**Requierements:**
- Python 3.10.0 or higher
- Pytorch
- ultralytics
- OpenCV
- xformers

> **Note**: Tested under Ubuntu 20.04 with Python 3.9.18

## Install miniconda (highly-recommended)
It is highly recommended to install all the dependencies on a new virtual environment. For more information check the conda documentation for [installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and [environment management](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). For creating the environment use the following commands on the terminal.

```bash
conda create -n User-guided python=3.9.18 numpy=1.24.2
conda activate User-guided
```

## Install from source
Firstly, clone the repository in your system.
```bash
git clone https://github.com/albmendez/User-guided-scene-with-StableDiffusion
```

Then, enter the directory and install the required dependencies
```bash
cd User-guided
pip install -r requirements.txt
```

# Usage
To use the TAICHI algorithm directly on the model of the robot and robotic arm presented in the paper, it is necessary to have both [MATLAB (R2022a)](https://es.mathworks.com/products/new_products/release2022a.html) and the [MujoCo](https://github.com/openai/mujoco-py) simulator installed. The simulator used in this work has been developed by our laboratory and all the information can be found in [ADAM Simulator](https://github.com/vistormu/adam_simulator). The algorithm has been tested and uses the specific packages for the [RealSense D435i](https://www.intelrealsense.com/depth-camera-d435i/) camera. Feel free to use our algorithm to applied it in other models and robotic arms.

The code is divided in two files:

`MatlabCode` stores all the files to apply the algorithm directly in Matlab.

`PythonCode` stores all the files to apply the algorithm directly in Python.

### **MatlabCode**
*Matlab Requierements:*
- Robotics System Toolbox
- Phased Array System Toolbox

The code available in **MatlabCode** contains:

`bodyx.stl` allows to load the full model of the ADAM robot in different body parts.

`HumanLikeArm.m` allows to run the full process once the data has been acquired.

`IK_UR.m` allows calculate the Analytical Inverse Kinematics (AIK) for different UR models such as UR3, UR5 and UR10.

If you want to use different UR model, in `IK_UR.m` modify the variables for the line:

```matlab
res = double(pyrunfile("UR3_Inverse_Kinematics.py","Sol",M = npM,name = name,launcher = launcher));
```
where `name` can be `ur3`, `ur5` or `ur10` and `launcher` can be `matlab` or `python`. The previous line calls `UR3_Inverse_Kinematics.py` that obtain the AIK values.

`distanceMetric.m` function to evaluate the distance between the human elbow and the rest of the robotic joints.

`distOrientation` function to evaluate the orientation of the end effector.

`distPosition` function to evaluate the position of the end effector.

`variationWrist` function to evaluate the wrist variation between the previous state and the actual state.

`PlotError.m` function to plot the error for the end-effector respect the human wrist.

To use the **MatlabCode** just run the `HumanLikeArm.m` script modifying the values of the path were the data is stored.

```matlab
path = '/your_directory/HumanLikeCode/HumanData/PruebaX/DatosBrazoHumano.csv';
path2 = '/your_directory/HumanLikeCode/HumanData/PruebaX/CodoHumano.csv';
```
### **PythonCode**
The code available in **PythonCode** contains:

`HumanLikeArm.py` works exactly the same as the MatlabCode one but all in Python.

`UR3_Inverse_Kinematics.py` allows to obtains the AIK solutions. The script return 8 solution for each point taking into account the limits for each arm.

`brazoProf.py` acquired the data from the camera an save it in the path specified. Must be the same that you use in `HumanLikeArm.py` and 
`HumanLikeArm.m`.

To use the **PythonCode** just run the `HumanLikeArm.py` script modifying the values of the path were the data is stored.

``` bash
cd Taichi/PythonCode
python HumanLikeArm.py
```

### **Data Acquisition**
The acquisition of the data can be only done using Python. For that purpose, you have to run the `brazoProf.py`:

``` bash
cd Taichi/PythonCode
python brazoProf.py
```
This will open a window that show the user moving with the MediaPipe lines for the Pose estimation and Hand Tracking.

<p align="center">
  <img src="./Images/DataCamera.png" height=200 />
</p>

Once the data as been acquired, pushing `q` or `esc` allows to stop the data acquisition. After that and previously to the storage of the data, a Gaussian Filter is applied. The filtered can be seen for wrist and elbow position and wrist orientation using:

``` python
plot_smoothed_EndEfector(DATOSPRE,XEnd,YEnd,ZEnd)
plot_smoothed_Elbow(CORCODOPRE,XElbow,YElbow,ZElbow)
plot_smoothed_rotations(DATOSPRE,R1,R2,R3,R4,R5,R6,R7,R8,R9)
```

<p align="center">
  <img src="./Images/Gaussian.png" height=200 />
</p>

> **Note**: At the moment, the algorithm only works for the left arm. To obtain correct data, the right arm must be hide.

# Citation
In process
