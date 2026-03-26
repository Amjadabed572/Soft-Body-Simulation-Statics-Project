# Animation and Robotics - Assignment 2: <br> Soft body simulation - Statics

## Introduction
A mass-spring system is one of the simplest models for an elastic object, yet implementing a simulator for such a system provides a solid foundation for understanding more complex models. Most simulators rely on the same building blocks regardless of the model, making the transition to sophisticated models more about changing a few formulas.

In this assignment, you will implement a static mass-spring system simulation. Static here means the simulator will only compute the steady-state, motionless configuration. You will learn about the common classes used in simulation and create a minimal GUI for interacting with the simulator.

## Instructions

### Preliminary steps:
1. Ensure you have completed the setup steps from Assignment 1, including the installation of Python, VS Code, and required extensions.
2. Make sure you have cloned the repository and have the environment set up correctly.

### Setup steps:
1. Create a folder with no spaces and no non-english characters (Preferably in `C:\Courses\AnimationAndRobotics\Assignments\`) and clone the assignment repository with `git clone`:

    `git clone https://github.com/xCraftCourses/animation-and-robotics-soft-body-simulation-statics-[your github id]`
    
    This will create a new folder that contains this repository.
2. Open the folder with VS Code.
3. Create a new Python environment (`CTRL-SHIFT-P`, type `python env` and select `Python: Create Environment`). Follow the steps. VS Code should create a new folder called `.venv`.
4. Open a new terminal (`` CTRL-SHIFT-` ``). If VS Code detected the Python environment correctly, the prompt should begin with `(.venv)`. If not, restart VS Code and try again. If it still doesn't make sure the default terminal is `cmd` or `bash` (use `CTRL-SHIFT-P` and then `Terminal: Select Default Profile` to change it) and start a new terminal. If it still fails, ask for help.
5. Install Vedo, a scientific visualization package for python, using `pip install vedo` in the terminal.
6. Open `Assignment2.py`. The file is divided into cells, where each cell is defined by `#%%`. Run the first cell. Recall that VS Code will tell you that it needs to install the ipykernel runtime. Make sure it ran without any errors.

## Introduction

The initial code for the assignment comes with a few basic classes to help you organize. These types of classes are commonly found in simulator code, but other structures and variation exist. They are roughly divided to classes that are related to the definition of the mesh and its energies, and the solver, also called optimizer. If you inspect the code, you will see a class called `FEMMesh`, which stores the topology of the mesh and its rest pose X and can compute the energy (and gradient and Hessian) given a deformed pose x. The energy is computed by itereating over all of the elements (edge, triangles, etc.) and summing up all of their energies. The computation of energy of each element are delegated to a different class which inherits from an abstract ElementEnergy class. Instances of this class are expected to implement an `energy` method, and can optionally implement a gradient and Hessian methods. If the gradient and Hessian are not implemented, the computation fallsback on the base class' finite difference computation.

The solver is comprised of several methods. The most front-facing one is `optimize`, which iteratively calls `step` to improve the solution. `step` consists of a `SearchDirection` method, and a `LineSearch` method, both of which point to one of several options. In our case, `GradientDescent` and `Newton` are implemented for the `SearchDirection` and only `BacktrackingLineSearch` is implemented for the line search. The compute the search direction, the solver simply calls the energy, gradient and Hessian methods of the FEMMesh class, and updates the deformed pose accordingly, using the line search function to insure a decrease in every iteration.

## Tasks
You are required to write a report in a markdown format (`.md`). The report will be fetched and processed automatically, so it is important you follow these steps:
1. Create an .md file with the work `report` in its name (e.g. `report.md`) and put it in the root path of the repository.
2. Put your full name and ID number somewhere in the report.

The report should be divided into tasks. In each part, use images and videos, with extra text to complement them, to show what you did. The images and videos should be added to the repository. They should clearly and *concisely* demonstrate the tasks were performed correctly. Avoid adding too many images, or videos longer than 5 seconds, as this does not add value to the report and in fact can even make it worse. 

### Task 1: Create a mesh for the simulation
The first step is to generate a mesh to use with the solver. To create the mesh, we will use the `triangle` package, which triangulates the interior of a polygon. Go over the sample code in the `Main program` cell. The line
```
vertices = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) # square
```
creates the polygon, followed by the line
```
tris = tr.triangulate({"vertices":vertices[:,0:2]}, f'qa0.01') # triangulate the square
```
which triangulates the interior. The lines
```
V = tris['vertices'] # get the vertices of the triangles
F = tris['triangles'] # get the triangles
mesh = vd.Mesh([V,F]).linecolor('black')
```
extract the faces and vertices and creates a new Vedo Mesh.

1. Create a different shape, other than the square in the sample code. Show it in the report. Be creative!
2. Read the documentation of Triangle and figure out how to create a triangulation with no interior vertices and with approximately 100 vertices. Show the results in the report.

### Task 2: Implement Basic UI for Moving Points
In order to interact with the simulator, we need to implement a basic UI. A simple option is to allow the user to *pin* vertices by clicking on them, and set a new location for pinned vertices. The lines
```
def OnLeftButtonPress(event):
    if event.object is None:          # mouse hits nothing, return.
        print('Mouse hits nothing')
    if event.object is mesh:          # mouse hits the mesh
        print('Mouse hits the mesh')
        print(event.picked3d)
```
demonstrate a simple interaction with the mesh. Note that the prints will only occur after the window is closed. 

1. Instead of printing to Jupyter (IPython), change the code such that it prints to the Vedo Window. This will make it easier to debug problems. Show an image in the report
2. Building on top of the basic functionality shown above, implement a mechanism that would allow the user to pin vertices and move pinned vertices to new locations. There are countless approaches for doing that. Make your own, and be creative! Remember you can use keyboard modifiers (Shift, CTRL, ALT, etc.) to change what happens when you click on an object with the mouse. Explain your interface in the report and add images/clips that could illustrate it if necessary.
3. Allow the user to switch between 2D and 3D modes, and to reposition vertices in 3D. Again, there are many ways of enabling this. Explain and demonstrate what you did in the report.

### Task 3: Test the optimization pipeline
1. Without any pinned vertices, create a FEMMesh with your mesh and the `ZeroLengthSpringEnergy` energy, and an optimizer by calling
```
femMesh = FEMMesh(mesh, ZeroLengthSpringEnergy)
optimizer = MeshOptimizer(femMesh)
```
Run several optimization steps by calling `x = optimizer.step(x)` and show the result after each iteration. What do you expect to see? Compare gradient descent and Newton's method and report on your findings.
2. Try to do the same with `SpringEnergy` and show the result in the report. Note that since the gradient and Hessian methods are not implemented, they are computed using finite differences.
3. Enable pinning vertices to specific locations using the UI from Task 2 and using soft constraints. Allow the user to modify the weights of the constraints in the GUI and show how the results differ for different values in the report.
4. Implement the missing gradient and Hessian of `SpringEnergy` and compare the performance of the analytical derivatives with the numerical (finite differences) derivative.

### Task 4: Collisions
1. Allow the user to define a ground and propose and implement an energy that will prevent vertices from penetrating the ground. Also allow the user to adjust the weight of this energy. Show the effect of including this energy in the simulation.
2. Allow the user to add a ball that will act as a collider. As in the previous task, suggest an energy that will prevent vertices from penetrating the ball, and demonstrate it in the report.

## Submission
The last commit before the deadline is the one that will be examined. Later commits will be ignored.

## Grading
Grading will be based on the following criteria:
* Correctness of the results
* Quality of the report
* Creativity
* Effort
* Commit history

The feedback will appear on the feedback pullrequest of your repository.
