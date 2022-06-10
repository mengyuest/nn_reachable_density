'''
Variable Symbol               Meaning
x[0]       Vt                 Air Speed
x[1]       α                  Angle of Attack
x[2]       β                  Angle of Side slip
x[3]       φ                  Roll
x[4]       θ                  Pitch
x[5]       ψ                  Yaw
x[6]       P                  Roll Rate
x[7]       Q                  Pitch Rate
x[8]       R                  Yaw Rate
x[9]       Pn                 Northward Displacement
x[10]      Pe                 Eastward Displacement
x[11]      alt                Altitude
x[12]      pow                Engine Power Lag
x[13]      Nz(integrator)     Upward Accel
x[14]      Ps(integrator)     Stability Roll Rate
x[15]      Ny + r(integrator) Side Accel and Yaw Rate
'''
import math
from numpy import deg2rad
import math
# TODO (change!)
# x[0]       Vt                 Air Speed
vt_min=540
vt_max=540

# TODO (change!)
# x[1]       α                  Angle of Attack
alpha_min=0.075
alpha_max=0.1

# x[2]       β                  Angle of Side slip
beta_min=0
beta_max=0

# TODO (change!)
# x[3]       φ                  Roll
phi_min=-0.1
phi_max=-0.075

# TODO (change!)
# x[4]       θ                  Pitch
theta_min=-math.pi/2*0.3
theta_max=-math.pi/2*0.3

# x[5]       ψ                  Yaw
psi_min=0
psi_max=0

# x[6]       P                  Roll Rate
p_min=0
p_max=0

# TODO (change!)
# x[7]       Q                  Pitch Rate
q_min=0
q_max=0

# x[8]       R                  Yaw Rate
r_min=0
r_max=0

# x[9]       Pn                 Northward Displacement
pn_min=0
pn_max=0

# x[10]      Pe                 Eastward Displacement
pe_min=0
pe_max=0

# TODO (change!)
# x[11]      alt                Altitude
alt_min=1000
alt_max=1000

# x[12]      pow                Engine Power Lag
power_min=9
power_max=9

# x[13]      Nz(integrator)     Upward Accel
nz_min=0
nz_max=0

# x[14]      Ps(integrator)     Stability Roll Rate
ps_min=0
ps_max=0

# x[15]      Ny + r(integrator) Side Accel and Yaw Rate
ny_min=0
ny_max=0

