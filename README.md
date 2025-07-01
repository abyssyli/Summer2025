# Update 0626
You can try either koffi062601.py and koffi 062602.py. I cannot use google colab currently but I will be back to the U.S. this weekend. If your cpu is not large enough I guess the final comparison plot would not show. That happened to my laptop twice this night which was fucking stupid. So I said please run the file on your laptop if it is avaliable. Thank you very much professor and see you after your lecture ends today!

# Summer2025

This repository was created to provide updates to Professor Koffi.

koffi610.py: The current version includes a notebook that uses Physics-Informed Neural Networks (PINNs) to solve the one-dimensional heat equation.

koffi611.py: This version added x/y axis labels with physical meaning and units (position in m, time in s), implemented boundary residual diagnostics, visualized and analyzed individual loss components ($L_f$, $L_i$, $L_b$), and verified PINN consistency during training.

koffi612.py: PINNS for waive equation.

koffi612.py issue: why the last graph is so wierd? Because there should be \lambda1, \lambda2, \lambda3, respectively, in front of each of the three loss funtions.

The next step: the reaction difussion equation. Due by next Tuesday 3 PM, LA time.

Update 0701: Removed the two .py files. Added the notebook for Linear Reaction Diffusion Equation. However, that is uselsss, because the linearity makes accurate approximation especially sensitive to the initial and boundary conditions, as any small mismatch propagates clearly through the solution.